# SPDX-License-Identifier: Apache-2.0
#
# Measurement environment for SOL-ExecBench-ROCm on CDNA4 (gfx950).
#
# Serves both parts: MI350X (air, 1000 W) and MI355X (liquid, 1400 W) are the
# same die, so they share the image and share nothing that was measured.
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
# pybind11: builds the rocprofiler-sdk activity shim (task 04).
RUN pip install --no-cache-dir uv pybind11

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

# SOLAR — the analytic SOL model (task 03). Not vendored: it is a separate
# Apache-2.0 repo (NVlabs/SOLAR) with its own release cadence, and the AMD port
# adds an arch config to it rather than changing it. Pinned to a SHA for the
# same reason every other version here is pinned: T_SOL must be reproducible,
# and a silently newer SOLAR would move every bound.
#
# `--no-deps` for solar itself plus an explicit, constrained dependency list:
# its requirements.txt asks for `torch>=2.0.0`, which pip would happily satisfy
# by replacing the ROCm build with a CUDA wheel from PyPI. Nothing would fail
# until the numbers did.
#
# SOLAR also requires a PATCHED torchview: stock torchview does not emit
# parameter-tensor nodes, so weights would be invisible to the analyzer and
# every weight-heavy problem's memory bound would come out too low -- SOLAR
# refuses to run rather than let that happen.
#
# SOLAR ships TWO patches. Only one is applied here, and the reason is
# checked rather than assumed:
#
#   torchview-parameter-tensors.patch   applied. Required.
#   torchview-collect-attributes.patch  NOT applied: both of its changes are
#       ALREADY PRESENT in torchview at the pinned commit (expressed with
#       functools.partial instead of the patch's lambda). The patch file is
#       also malformed -- its first hunk header claims 9 lines and supplies 8
#       -- so `git apply` refuses it, and SOLAR's own install.sh silently
#       skips it on failure. Silently skipping a patch that mattered would
#       leave reduction-op attributes (dim/keepdim) uncaptured and quietly
#       change the analysis, so the presence of both changes is ASSERTED
#       below. If a future torchview bump drops them, this build fails
#       instead of producing subtly wrong SOL bounds.
ARG SOLAR_SHA=d3524c4488f32db002d5f1e33b5d58d01b707b51
ARG TORCHVIEW_COMMIT=edbe1fa
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        networkx pyyaml matplotlib scipy pandas graphviz \
 && git clone -q https://github.com/mert-kurttutan/torchview.git /tmp/torchview \
 && git -C /tmp/torchview checkout -q ${TORCHVIEW_COMMIT} \
 && git clone -q https://github.com/NVlabs/SOLAR /tmp/solar-src \
 && git -C /tmp/solar-src checkout -q ${SOLAR_SHA} \
 && git -C /tmp/torchview apply /tmp/solar-src/patches/torchview-parameter-tensors.patch \
 && python -c "import pathlib; \
tv = pathlib.Path('/tmp/torchview/src/torchview'); \
a = (tv/'torchview.py').read_text(); r = (tv/'recorder_tensor.py').read_text(); \
assert a.count('get_recorder_tensor, collect_attributes=collect_attributes') >= 2, \
    'torchview lost collect_attributes propagation in process_input'; \
assert r.count(chr(39)+'collect_attributes'+chr(39)+': collect_attributes') >= 2, \
    'torchview lost collect_attributes in attach_kwargs'; \
print('collect-attributes changes verified present upstream')" \
 && pip install --no-cache-dir --no-deps /tmp/torchview \
 && pip install --no-cache-dir --no-deps /tmp/solar-src \
 && rm -rf /tmp/torchview /tmp/solar-src \
 && python -c "import torch, solar, torchview; \
               assert 'rocm' in torch.__version__, torch.__version__; \
               print('torch intact after SOLAR:', torch.__version__)"

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
