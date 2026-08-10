import torch


@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    # Keep the singleton KV-head dimension and let matmul broadcast it across
    # query heads.  This avoids materializing four identical copies of K.
    scores = torch.matmul(
        query.to(torch.float32), key.transpose(2, 3).to(torch.float32)
    )
    return (scores * scaling).to(query.dtype)
