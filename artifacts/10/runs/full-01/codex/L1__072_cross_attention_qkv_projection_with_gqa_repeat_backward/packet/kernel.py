import torch
import triton
import triton.language as tl


def _eager_preprocess(grad_query_states, grad_key_states, grad_value_states):
    batch_size = grad_query_states.shape[0]
    seq_len_dec = grad_query_states.shape[2]
    seq_len_enc = grad_key_states.shape[2]
    q = grad_query_states.transpose(1, 2).contiguous().view(
        batch_size * seq_len_dec, 4096
    )
    k = (
        grad_key_states.view(batch_size, 4, 4, seq_len_enc, 256)
        .sum(dim=2)
        .transpose(1, 2)
        .contiguous()
        .view(batch_size * seq_len_enc, 1024)
    )
    v = (
        grad_value_states.view(batch_size, 4, 4, seq_len_enc, 256)
        .sum(dim=2)
        .transpose(1, 2)
        .contiguous()
        .view(batch_size * seq_len_enc, 1024)
    )
    return q, k, v


@torch.compile(fullgraph=True, dynamic=True)
def _compiled_preprocess(grad_query_states, grad_key_states, grad_value_states):
    return _eager_preprocess(grad_query_states, grad_key_states, grad_value_states)


@triton.jit
def _small_preprocess_kernel(
    grad_q,
    grad_k,
    grad_v,
    q_out,
    kv_out,
    Q_PROGRAMS: tl.constexpr,
    KV_ROWS: tl.constexpr,
    SEQ_DEC: tl.constexpr,
    SEQ_ENC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if pid < Q_PROGRAMS:
        row = pid // 4
        chunk = pid % 4
        col = chunk * BLOCK + lane
        batch = row // SEQ_DEC
        seq = row % SEQ_DEC
        head = col // 256
        dim = col % 256
        src = ((batch * 16 + head) * SEQ_DEC + seq) * 256 + dim
        value = tl.load(grad_q + src)
        tl.store(q_out + row * 4096 + col, value)
    else:
        kv_pid = pid - Q_PROGRAMS
        plane = kv_pid // KV_ROWS
        row = kv_pid % KV_ROWS
        batch = row // SEQ_ENC
        seq = row % SEQ_ENC
        head = lane // 256
        dim = lane % 256
        src = ((batch * 16 + head * 4) * SEQ_ENC + seq) * 256 + dim
        group_stride = SEQ_ENC * 256
        if plane == 0:
            x0 = tl.load(grad_k + src)
            x1 = tl.load(grad_k + src + group_stride)
            x2 = tl.load(grad_k + src + 2 * group_stride)
            x3 = tl.load(grad_k + src + 3 * group_stride)
        else:
            x0 = tl.load(grad_v + src)
            x1 = tl.load(grad_v + src + group_stride)
            x2 = tl.load(grad_v + src + 2 * group_stride)
            x3 = tl.load(grad_v + src + 3 * group_stride)
        value = ((x0 + x1) + x2) + x3
        tl.store(kv_out + plane * KV_ROWS * 1024 + row * 1024 + lane, value)


def _small_preprocess(grad_query_states, grad_key_states, grad_value_states):
    batch_size = grad_query_states.shape[0]
    seq_len_dec = grad_query_states.shape[2]
    seq_len_enc = grad_key_states.shape[2]
    query_rows = batch_size * seq_len_dec
    kv_rows = batch_size * seq_len_enc
    q = torch.empty(
        (query_rows, 4096), device=grad_query_states.device, dtype=torch.float32
    )
    kv = torch.empty(
        (2, kv_rows, 1024), device=grad_query_states.device, dtype=torch.float32
    )
    q_programs = query_rows * 4
    _small_preprocess_kernel[(q_programs + 2 * kv_rows,)](
        grad_query_states,
        grad_key_states,
        grad_value_states,
        q,
        kv,
        Q_PROGRAMS=q_programs,
        KV_ROWS=kv_rows,
        SEQ_DEC=seq_len_dec,
        SEQ_ENC=seq_len_enc,
        BLOCK=1024,
        num_warps=4,
    )
    return q, kv


@torch.no_grad()
def run(
    grad_query_states,
    grad_key_states,
    grad_value_states,
    decoder_hidden_states,
    encoder_hidden_states,
    q_weight,
    k_weight,
    v_weight,
):
    batch_size, seq_len_dec, _ = decoder_hidden_states.shape
    seq_len_enc = encoder_hidden_states.shape[1]
    query_rows = batch_size * seq_len_dec
    kv_rows = batch_size * seq_len_enc
    use_batched_weights = kv_rows <= 6000
    if batch_size == 4 and seq_len_dec == 2048 and seq_len_enc == 4096:
        grad_query_proj, grad_key_proj, grad_value_proj = _compiled_preprocess(
            grad_query_states, grad_key_states, grad_value_states
        )
    else:
        grad_query_proj, grad_kv = _small_preprocess(
            grad_query_states, grad_key_states, grad_value_states
        )
        grad_key_proj, grad_value_proj = grad_kv[0], grad_kv[1]

    decoder_flat = decoder_hidden_states.view(query_rows, 1536)
    encoder_flat = encoder_hidden_states.view(kv_rows, 1024)

    grad_decoder_hidden_states = torch.mm(grad_query_proj, q_weight).view(
        batch_size, seq_len_dec, 1536
    )
    grad_q_weight = torch.mm(grad_query_proj.t(), decoder_flat)

    grad_encoder_hidden_states = torch.mm(grad_key_proj, k_weight)
    grad_encoder_hidden_states.addmm_(grad_value_proj, v_weight)
    grad_encoder_hidden_states = grad_encoder_hidden_states.view(
        batch_size, seq_len_enc, 1024
    )

    if use_batched_weights:
        encoder_batch = encoder_flat.unsqueeze(0).expand(2, -1, -1)
        grad_kv_weight = torch.bmm(grad_kv.transpose(1, 2), encoder_batch)
        grad_k_weight, grad_v_weight = grad_kv_weight[0], grad_kv_weight[1]
    else:
        grad_k_weight = torch.mm(grad_key_proj.t(), encoder_flat)
        grad_v_weight = torch.mm(grad_value_proj.t(), encoder_flat)

    return (
        grad_decoder_hidden_states,
        grad_encoder_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
    )
