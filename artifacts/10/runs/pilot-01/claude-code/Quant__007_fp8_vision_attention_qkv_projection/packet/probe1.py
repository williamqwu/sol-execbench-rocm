import torch, inspect
print(torch.__version__)
print("has _scaled_mm:", hasattr(torch, "_scaled_mm"))
try:
    print(torch._scaled_mm.__doc__[:2000])
except Exception as e:
    print("doc err", e)
try:
    import torch._C
    print(torch.ops.aten._scaled_mm.default._schema)
except Exception as e:
    print("schema err", e)
try:
    print(torch.ops.aten._scaled_mm_v2.default._schema)
except Exception as e:
    print("v2 err", e)
try:
    print(torch.ops.aten._scaled_grouped_mm.default._schema)
except Exception as e:
    print("gg err", e)
