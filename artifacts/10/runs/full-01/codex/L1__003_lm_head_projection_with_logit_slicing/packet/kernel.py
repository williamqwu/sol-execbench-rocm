import torch
import aiter


# AITER's hipBLASLt solution API exposes the precompiled gfx950 Tensile
# kernels.  Initialize its handle once; individual calls below remain ordinary
# GEMMs and never retain inputs or outputs.
aiter.hipb_create_extension()
_hipb_mm = torch.ops.aiter.hipb_mm.default


def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    batch, seq_len, hidden_size = hidden_states.shape
    m = batch * seq_len

    # These are the fastest solution families for the fixed N=102400,
    # K=2048 projection.  Ranges correspond to changes in the preferred M
    # macro-tile, rather than to input values.
    if m <= 128:
        solution = 439338
    elif m <= 256:
        solution = 438248
    elif m <= 384:
        solution = 439421
    elif m <= 3584:
        solution = 440238
    else:
        solution = 438241

    x = hidden_states.view(m, hidden_size)
    logits = _hipb_mm(x, weight.t(), solution)
    return logits.view(batch, seq_len, weight.shape[0])
