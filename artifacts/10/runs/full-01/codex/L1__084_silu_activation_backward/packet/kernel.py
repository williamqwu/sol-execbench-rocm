import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256, "PERSIST": 0, "GRID_MULT": 1}, num_warps=4),
        triton.Config({"BLOCK": 512, "PERSIST": 0, "GRID_MULT": 1}, num_warps=4),
        triton.Config({"BLOCK": 1024, "PERSIST": 0, "GRID_MULT": 1}, num_warps=4),
        triton.Config({"BLOCK": 1024, "PERSIST": 0, "GRID_MULT": 1}, num_warps=8),
        triton.Config({"BLOCK": 2048, "PERSIST": 0, "GRID_MULT": 1}, num_warps=4),
        triton.Config({"BLOCK": 2048, "PERSIST": 0, "GRID_MULT": 1}, num_warps=8),
        triton.Config({"BLOCK": 4096, "PERSIST": 0, "GRID_MULT": 1}, num_warps=8),
        triton.Config({"BLOCK": 1024, "PERSIST": 1, "GRID_MULT": 1}, num_warps=8),
        triton.Config({"BLOCK": 1024, "PERSIST": 1, "GRID_MULT": 2}, num_warps=8),
        triton.Config({"BLOCK": 1024, "PERSIST": 1, "GRID_MULT": 4}, num_warps=8),
        triton.Config({"BLOCK": 1024, "PERSIST": 1, "GRID_MULT": 8}, num_warps=8),
        triton.Config({"BLOCK": 2048, "PERSIST": 1, "GRID_MULT": 1}, num_warps=8),
        triton.Config({"BLOCK": 2048, "PERSIST": 1, "GRID_MULT": 2}, num_warps=8),
        triton.Config({"BLOCK": 2048, "PERSIST": 1, "GRID_MULT": 4}, num_warps=8),
        triton.Config({"BLOCK": 2048, "PERSIST": 1, "GRID_MULT": 8}, num_warps=8),
        triton.Config({"BLOCK": 4096, "PERSIST": 1, "GRID_MULT": 2}, num_warps=8),
        triton.Config({"BLOCK": 4096, "PERSIST": 1, "GRID_MULT": 4}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _silu_backward_kernel(
    grad_output_ptr,
    x_ptr,
    sigmoid_ptr,
    grad_input_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
    PERSIST: tl.constexpr,
    GRID_MULT: tl.constexpr,
):
    pid = tl.program_id(0)
    lane_offsets = tl.arange(0, BLOCK)

    if PERSIST:
        tile = pid
        num_tiles = tl.cdiv(n_elements, BLOCK)
        tile_stride = tl.num_programs(0)
        while tile < num_tiles:
            offsets = tile * BLOCK + lane_offsets
            mask = offsets < n_elements
            grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
            x = tl.load(x_ptr + offsets, mask=mask)
            sigmoid_x = tl.load(sigmoid_ptr + offsets, mask=mask)
            one_minus_sigmoid = 1.0 - sigmoid_x
            x_times_one_minus_sigmoid = x * one_minus_sigmoid
            bracket_term = 1.0 + x_times_one_minus_sigmoid
            local_grad = sigmoid_x * bracket_term
            grad_input = grad_output * local_grad
            tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
            tile += tile_stride
    else:
        offsets = pid * BLOCK + lane_offsets
        mask = offsets < n_elements
        grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        sigmoid_x = tl.load(sigmoid_ptr + offsets, mask=mask)
        one_minus_sigmoid = 1.0 - sigmoid_x
        x_times_one_minus_sigmoid = x * one_minus_sigmoid
        bracket_term = 1.0 + x_times_one_minus_sigmoid
        local_grad = sigmoid_x * bracket_term
        grad_input = grad_output * local_grad
        tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    sigmoid_x: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(grad_output)
    n_elements = grad_output.numel()
    def grid(meta):
        tiles = triton.cdiv(n_elements, meta["BLOCK"])
        if meta["PERSIST"]:
            return (min(tiles, 256 * meta["GRID_MULT"]),)
        return (tiles,)

    _silu_backward_kernel[grid](
        grad_output,
        x,
        sigmoid_x,
        output,
        n_elements=n_elements,
    )
    return output
