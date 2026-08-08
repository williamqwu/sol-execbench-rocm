import torch
import triton
import triton.language as tl

@triton.jit
def _inv_freq_kernel(out_ptr, rope_theta, HEAD_DIM: tl.constexpr):
    offs = tl.arange(0, HEAD_DIM // 2)
    theta = rope_theta.to(tl.float64)
    exps = (2.0 * offs.to(tl.float64)) / float(HEAD_DIM)
    powers = tl.exp(exps * tl.log(theta))
    inv = (1.0 / powers).to(tl.float32)
    tl.store(out_ptr + offs, inv)

_warmup_out = torch.empty(64, dtype=torch.float32, device='cuda')
_inv_freq_kernel[(1,)](_warmup_out, 1000000.0, HEAD_DIM=128)
torch.cuda.synchronize()

_dev_cache = _inv_freq_kernel.device_caches[0]
_ck = list(_dev_cache[0].values())[0]
_launcher = _ck.run
_launch_fn = _launcher.launch
_function = _ck.function
_kernel_metadata = (
    _ck.metadata.num_warps, 1, _ck.metadata.shared,
    _ck.metadata.cluster_dims[0], _ck.metadata.cluster_dims[1], _ck.metadata.cluster_dims[2],
)
_lcgrid = _launcher.launch_cooperative_grid
_stream = torch.cuda.current_stream().cuda_stream
_HEAD_DIM = 128

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty(64, dtype=torch.float32, device='cuda')
    _launch_fn(
        _lcgrid, 1, 1, 1, _stream, _function, None,
        _kernel_metadata, None, None, None,
        out.data_ptr(), float(rope_theta), _HEAD_DIM
    )
    return out
