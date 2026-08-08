import triton, dataclasses
from triton.backends.amd.compiler import HIPOptions
for f in dataclasses.fields(HIPOptions):
    print(f.name, '=', f.default)
