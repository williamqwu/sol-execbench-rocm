import math
import torch

DIM = 1536
CDIM = 1152
FF = 6144

SHAPES = {
    "hidden_states": ("B", "S", DIM),
    "encoder_hidden_states": ("B", "C", CDIM),
    "temb": ("B", DIM),
    "norm1_weight": (6 * DIM, DIM),
    "norm1_bias": (6 * DIM,),
    "norm1_context_weight": (6 * CDIM, DIM),
    "norm1_context_bias": (6 * CDIM,),
    "to_q_weight": (DIM, DIM),
    "to_q_bias": (DIM,),
    "to_k_weight": (DIM, DIM),
    "to_k_bias": (DIM,),
    "to_v_weight": (DIM, DIM),
    "to_v_bias": (DIM,),
    "add_q_proj_weight": (DIM, CDIM),
    "add_q_proj_bias": (DIM,),
    "add_k_proj_weight": (DIM, CDIM),
    "add_k_proj_bias": (DIM,),
    "add_v_proj_weight": (DIM, CDIM),
    "add_v_proj_bias": (DIM,),
    "to_out_weight": (DIM, DIM),
    "to_out_bias": (DIM,),
    "to_add_out_weight": (CDIM, DIM),
    "to_add_out_bias": (CDIM,),
    "ff_linear1_weight": (FF, DIM),
    "ff_linear1_bias": (FF,),
    "ff_linear2_weight": (DIM, FF),
    "ff_linear2_bias": (DIM,),
    "ff_context_linear1_weight": (FF, CDIM),
    "ff_context_linear1_bias": (FF,),
    "ff_context_linear2_weight": (CDIM, FF),
    "ff_context_linear2_bias": (CDIM,),
}

# Mirrors sol_execbench.core.bench.io heuristics for this problem's input names.
NORM_W = {"norm1_weight"}          # prefix "norm1" -> strip digits -> "norm"
NORM_B = {"norm1_bias"}
WEIGHT_MATRIX = {n for n in SHAPES if n.endswith(("_weight", "_proj_weight")) and len(SHAPES[n]) >= 2}


def gen(B, S, C, device, seed=0):
    torch.manual_seed(seed)
    out = {}
    for name, shp in SHAPES.items():
        shp = tuple({"B": B, "S": S, "C": C}.get(d, d) for d in shp)
        if name in NORM_W:
            t = torch.ones(shp, device=device)
        elif name in NORM_B:
            t = torch.zeros(shp, device=device)
        elif name in WEIGHT_MATRIX:
            t = torch.randn(shp, device=device) / math.sqrt(shp[-1])
        else:
            t = torch.randn(shp, device=device)
        out[name] = t
    return out
