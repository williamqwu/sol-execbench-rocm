import torch


@torch.no_grad()
def run(hidden_states, q_weight, k_weight, v_weight):
    batch_size, seq_len, _ = hidden_states.shape
    token_count = batch_size * seq_len

    # One wide projection removes two GEMM launches.  For the largest M, two
    # better-shaped GEMMs (Q and KV) are a little faster on hipBLASLt.
    if token_count < 8192:
        packed_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        qkv = torch.matmul(hidden_states, packed_weight.t())
        stride = (seq_len * 3072, 128, 3072, 1)
        return (
            qkv.as_strided((batch_size, 16, seq_len, 128), stride, 0),
            qkv.as_strided((batch_size, 4, seq_len, 128), stride, 2048),
            qkv.as_strided((batch_size, 4, seq_len, 128), stride, 2560),
        )

    packed_kv = torch.cat((k_weight, v_weight), dim=0)
    query = torch.matmul(hidden_states, q_weight.t())
    kv = torch.matmul(hidden_states, packed_kv.t())
    return (
        query.as_strided(
            (batch_size, 16, seq_len, 128),
            (seq_len * 2048, 128, 2048, 1),
            0,
        ),
        kv.as_strided(
            (batch_size, 4, seq_len, 128),
            (seq_len * 1024, 128, 1024, 1),
            0,
        ),
        kv.as_strided(
            (batch_size, 4, seq_len, 128),
            (seq_len * 1024, 128, 1024, 1),
            512,
        ),
    )
