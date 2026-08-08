import torch, triton, triton.language as tl
print("triton", triton.__version__, "torch", torch.__version__)
from triton.language.extra import libdevice as ld
print([n for n in dir(ld) if 'exp' in n or 'sig' in n or 'div' in n])
