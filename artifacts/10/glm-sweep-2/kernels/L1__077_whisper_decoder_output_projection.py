import os
import tempfile

import torch
import torch.cuda.tunable as _tunable


# Pre-tuned hipBLASLt/rocBLAS GEMM algorithm cache for the 16 evaluation
# shapes on MI350X (gfx950). TunableOp's default heuristic does not always pick
# the fastest algorithm for these (batch, seq) combinations; enumerating the
# algorithm space offline and shipping the winners lets every matmul use the
# best kernel without paying the tuning cost at runtime.
_TUNING_CACHE = """Validator,PT_VERSION,2.9.1
Validator,HIP_VERSION,702
Validator,HIPBLASLT_VERSION,100201-5b515cf1bc
Validator,GCN_ARCH_NAME,gfx950:sramecc+:xnack-
Validator,ROCBLAS_VERSION,5.2.0.5b515cf1bc
GemmTunableOp_Half_TN,tn_51866_39232_1280_ld_1280_1280_51866,Gemm_Hipblaslt_477479,4.87196
GemmTunableOp_Half_TN,tn_51866_1172_1280_ld_1280_1280_51866,Gemm_Rocblas_477326,0.185983
GemmTunableOp_Half_TN,tn_51866_2048_1280_ld_1280_1280_51866,Gemm_Rocblas_476662,0.301475
GemmTunableOp_Half_TN,tn_51866_7184_1280_ld_1280_1280_51866,Gemm_Rocblas_477480,0.968481
GemmTunableOp_Half_TN,tn_51866_16_1280_ld_1280_1280_51866,Gemm_Rocblas_477365,0.0287017
GemmTunableOp_Half_TN,tn_51866_4_1280_ld_1280_1280_51866,Gemm_Rocblas_477397,0.0288644
GemmTunableOp_Half_TN,tn_51866_4096_1280_ld_1280_1280_51866,Gemm_Hipblaslt_477479,0.561071
GemmTunableOp_Half_TN,tn_51866_17312_1280_ld_1280_1280_51866,Gemm_Hipblaslt_477479,2.04089
GemmTunableOp_Half_TN,tn_51866_8192_1280_ld_1280_1280_51866,Gemm_Rocblas_477480,1.04675
GemmTunableOp_Half_TN,tn_51866_1_1280_ld_1280_1280_51866,Gemm_Rocblas_477365,0.0286204
GemmTunableOp_Half_TN,tn_51866_128_1280_ld_1280_1280_51866,Gemm_Hipblaslt_477337,0.0371804
GemmTunableOp_Half_TN,tn_51866_512_1280_ld_1280_1280_51866,Gemm_Rocblas_477346,0.0957535
GemmTunableOp_Half_TN,tn_51866_256_1280_ld_1280_1280_51866,Gemm_Hipblaslt_477398,0.0542541
GemmTunableOp_Half_TN,tn_51866_422_1280_ld_1280_1280_51866,Gemm_Rocblas_477331,0.0864505
"""


def _enable_tuning() -> None:
    """Activate TunableOp with the shipped algorithm cache.

    Writes the embedded cache to a temp file and points TunableOp at it. The
    first matmul reads the cache (one-time, ~3s); subsequent calls dispatch
    directly to the cached algorithm. If anything goes wrong we silently fall
    back to the default hipBLASLt heuristic so the kernel still produces correct
    results.
    """
    try:
        fd, cache_path = tempfile.mkstemp(suffix=".csv", prefix="tunable_")
        with os.fdopen(fd, "w") as f:
            f.write(_TUNING_CACHE)
        os.environ["TORCH_TUNABLEOP_ENABLED"] = "1"
        os.environ["TORCH_TUNABLEOP_FILENAME"] = cache_path
        _tunable.set_filename(cache_path)
        _tunable.set_max_tuning_iterations(1)
        _tunable.set_max_tuning_duration(5)
        _tunable.enable(True)
    except Exception:
        pass


_enable_tuning()


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Whisper decoder output projection: projects hidden states to vocabulary logits.

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, d_model=1280)
        weight: Tensor of shape (vocab_size=51866, d_model=1280)

    Returns:
        logits: Tensor of shape (batch_size, seq_len, vocab_size=51866)
    """
    logits = torch.matmul(hidden_states, weight.t())
    return logits
