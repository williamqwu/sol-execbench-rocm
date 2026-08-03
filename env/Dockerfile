# SPDX-License-Identifier: Apache-2.0
#
# Measurement environment for SOL-ExecBench-AMD on MI355X.
#
# Pinned base. Do NOT bump the torch/ROCm tag to make something work: every
# measured baseline is relative to this stack (prime directive 6). If the pin is
# incompatible with something, record it in STATE.md and raise it.
FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

# amdsmi ships as source inside the base image and is NOT pre-installed, but
# task 00's per-GPU equivalence check and task 01's clock sampling both need it.
# Installing from /opt/rocm keeps it locked to the image's own ROCm version.
RUN pip install --no-cache-dir /opt/rocm/share/amd_smi

# huggingface_hub: dataset fetch.  pyarrow: the dataset ships as parquet.
RUN pip install --no-cache-dir huggingface_hub==1.26.0 pyarrow

# uv: upstream's end-to-end tests invoke the CLI through `uv run`.
RUN pip install --no-cache-dir uv

# Upstream sol-execbench runtime deps, MINUS the NVIDIA wheel set
# (cuda-bindings, cuda-python, cupti-python, cuda-tile, cupy-cuda13x,
# nvidia-cutlass-dsl, nvidia-cudnn-frontend, nvidia-libcal-cu12,
# nvidia-cupynumeric), which PLAN.md moves behind an optional [nvidia] extra.
#
# The constraint file is load-bearing: several of these declare a plain `torch`
# dependency, and without it pip would happily replace the ROCm build with a
# CUDA wheel from PyPI -- silently invalidating every measurement taken
# afterwards, on a box where nothing would obviously break until timings did.
RUN python - <<'PY' > /tmp/constraints.txt
import importlib.metadata as md
for p in ("torch", "numpy", "triton", "pytorch-triton-rocm"):
    try:
        print(f"{p}=={md.version(p)}")
    except Exception:
        pass
PY
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        einops numcodecs pydantic safetensors rich datasets \
        torch-c-dlpack-ext apache-tvm-ffi \
 && python -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; \
               print('torch intact:', torch.__version__)"

# The repo is bind-mounted from a host user's home; git refuses to read a
# repo it does not own, which silently nulls git_sha in every provenance
# block. An artifact without provenance is not usable for scoring.
RUN git config --global --add safe.directory /work

ENV PYTHONPATH=/work/src
WORKDIR /work

# Upstream declares `sol-execbench = "sol_execbench.cli:cli"` as a console
# script, and its end-to-end tests invoke it through `uv run sol-execbench`.
# The package is bind-mounted at runtime rather than pip-installed (the source
# does not exist at image build time, and the container runs unprivileged so it
# cannot install into /opt/venv later), so provide the entry point directly.
RUN printf '#!/bin/sh\nexec python -c "from sol_execbench.cli.main import cli; cli()" "$@"\n' \
      > /usr/local/bin/sol-execbench \
 && chmod 0755 /usr/local/bin/sol-execbench
