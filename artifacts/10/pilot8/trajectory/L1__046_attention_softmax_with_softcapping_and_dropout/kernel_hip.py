import os
import torch
from torch.utils.cpp_extension import load_inline
from csrc import CUDA_SRC

_dir = os.path.dirname(os.path.abspath(__file__))
_bd = os.path.join(_dir, ".ext_build")
os.makedirs(_bd, exist_ok=True)

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

_ext = load_inline(
    name="softcap_softmax_ext",
    cuda_sources=CUDA_SRC,
    cpp_sources="at::Tensor run(const at::Tensor& x_);\nat::Tensor softcap_dbg(const at::Tensor& x);",
    functions=["run", "softcap_dbg"],
    extra_cuda_cflags=["-O3", "-ffp-contract=fast", "--offload-arch=gfx950"],
    build_directory=_bd,
    verbose=False,
)


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    return _ext.run(attn_weights)
