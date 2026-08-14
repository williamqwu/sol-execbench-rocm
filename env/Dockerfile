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

# aiter -- AMD's kernel library, and on this port a LANGUAGE an agent writes in.
#
# 29 of the harvested full-01 solutions (24 codex, 5 claude-code) `import aiter`.
# The node those agents ran on carried it at /sgl-workspace/aiter; this image
# never did. Re-scoring them here without it records 29 ImportErrors as model
# failures when they are environment failures -- and 24 of the 29 land on one
# harness, so the missing package would come out the other end as a claim about
# harness quality.
#
# This is an ADDITION, not a bump. Nothing already pinned moves, and the build
# asserts that below rather than trusting it.
#
# THE PIN is not chosen, it is recovered: every artifact in
# artifacts/10/scores/full-01/ stamps
# kernel_stack.aiter.git_sha = d9e5ef7ce08ee7045d583aed768cff41aa9210fe-dirty
# from host mia1-p02-g10, so this is the exact commit those 29 solutions were
# written and verified against.
#
# The `-dirty` is NOT reproduced and cannot be: that checkout carried
# uncommitted changes nobody recorded (aiter's own `pretune` rewrites tracked
# GEMM tables under hsa/, which is enough to do it). This image pins the
# committed tree, and env/check_stack.py refuses a dirty checkout so the
# difference stays visible instead of being inherited silently.
#
# TWO things in aiter's installer destroy the pinned stack at their defaults,
# and neither of them fails:
#   * setup.py runs .github/scripts/install_triton.sh whenever torch >= 2.9.1
#     -- ours is exactly 2.9.1 -- and that script begins
#     `pip uninstall -y triton pytorch-triton-rocm` before installing an
#     aiter-built wheel. A Triton kernel's performance is a property of the
#     compiler that built it (STATE.md D16), so that swap would invalidate every
#     Triton measurement in the repo. The constraint file cannot stop it because
#     it bypasses pip's resolver. AITER_USE_SYSTEM_TRITON=1 is the off switch;
#     the second check_stack.py call below is the proof it stayed off.
#   * in develop mode setup.py shells out to an UNCONSTRAINED
#     `pip install flydsl==0.2.4` unless that version is already present. It is
#     installed here first, under the same constraint file as everything else,
#     so the subprocess never runs.
#
# Installed EDITABLE, with the .git directory kept, for provenance not taste:
# scripts/provenance.py recovers aiter's SHA by running git in the parent of the
# import path, because aiter publishes no useful dist version. A wheel install
# would land in site-packages and stamp `git_sha: null` on every artifact -- for
# the one dependency that IS the kernel.
#
# SETUPTOOLS_SCM_PRETEND_VERSION carries the same SHA into the dist metadata.
# A shallow clone has no tags, so setuptools_scm would otherwise fall back to
# the "0.1.0" declared in aiter's pyproject -- the same string for every commit,
# i.e. a version that identifies nothing. This way the pin survives even if git
# refuses the directory or the .git tree is later stripped.
#
# Build cost, measured on this node: the pinned clone with its
# composable_kernel submodule is 433 MB and takes ~8 s. The pip step compiles
# nothing -- with PREBUILD_KERNELS unset aiter declares no ext_modules -- so it
# is metadata only. The compile cost is deferred to the GPU instead: aiter
# JIT-builds each op the first time it is called. That build goes to
# $HOME/.aiter/jit, because /opt/aiter is root-owned and aiter's
# get_user_jit_dir() copies its jit tree to $HOME when it cannot write beside
# the package. env/solb points HOME at the scratch mount, so the cache is
# persistent and shared between containers. That copytree is NOT locked (the
# per-module builds are), so the FIRST `import aiter` after a rebuild must be a
# single process -- warm it once before starting an 8-way shard sweep.
#
# safe.directory is --system, not --global: `git config --global` as root writes
# /root/.gitconfig, and the container runs as the invoking user with HOME
# redirected to the scratch mount, so that file is never read. /work escapes the
# problem only because the invoking user owns it. /opt/aiter is root-owned, so
# without a system-wide entry git calls it dubious ownership, provenance.py's
# git_sha returns None, and every artifact records `null` for the library that
# produced the kernel. The submodule needs its own entry because `git status`
# opens it.
COPY check_stack.py /opt/solbench/check_stack.py
ARG AITER_SHA=d9e5ef7ce08ee7045d583aed768cff41aa9210fe
RUN SOLB_STACK_RECORD=/opt/solbench/stack-pre-aiter.json \
      python /opt/solbench/check_stack.py \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
        flydsl==0.2.4 ninja psutil packaging pybind11 einops pandas \
 && git init -q /opt/aiter \
 && git -C /opt/aiter remote add origin https://github.com/ROCm/aiter \
 && git -C /opt/aiter fetch -q --depth 1 origin ${AITER_SHA} \
 && git -C /opt/aiter checkout -q FETCH_HEAD \
 && git -C /opt/aiter submodule update -q --init --depth 1 --recursive \
 && AITER_USE_SYSTEM_TRITON=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0+g${AITER_SHA} \
    pip install --no-cache-dir --no-deps -c /tmp/constraints.txt -e /opt/aiter \
 && git config --system --add safe.directory /opt/aiter \
 && git config --system --add safe.directory /opt/aiter/3rdparty/composable_kernel \
 && SOLB_WANT_AITER_SHA=${AITER_SHA} \
    SOLB_COMPARE_RECORD=/opt/solbench/stack-pre-aiter.json \
    SOLB_STACK_RECORD=/opt/solbench/stack.json \
      python /opt/solbench/check_stack.py

# Exported so the image states its own pin: `env/solb python env/check_stack.py`
# then asserts it with no arguments, and `docker inspect` shows which aiter an
# artifact from this image was measured against.
ENV SOLB_WANT_AITER_SHA=${AITER_SHA}

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
