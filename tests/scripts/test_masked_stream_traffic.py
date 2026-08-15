# SPDX-License-Identifier: Apache-2.0
"""D18's second guise: a masked STREAM, not a gathered ALLOCATION.

The declared-traffic tier charged a full read of `q` on the causal paged
prefill problems, where the reference's own `if max_kv_idx <= 0: continue`
leaves 25 of 15783, 2, 3 and 1 query rows live on the four workloads real
kernels falsified. Priced at the declared stream, those four bounds demanded
8.24-10.10 TB/s of a part measured to reach ~7.3 TB/s at that working-set
size; priced at the live rows they sit at 4.16-5.09 TB/s.

These tests pin the four places the rule can go wrong. It must fire on the
paged-prefill shape and derive the whole pairing without a problem name; it
must NOT fire on the neighbouring problems, which mask by `masked_fill_` and
emit NaN on an empty window rather than skipping (`FlashInfer-Bench__016/017`)
or run every row (`__019`); the live-row count must come out of the workload's
own index vectors and refuse a blob that does not match the declared axes; and
the OUTPUT must keep its full price, because a correct kernel still has to
write `(0, -inf)` into every dead row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sol_gathered_traffic import (causal_masked_axis,  # noqa: E402
                                  gathered_traffic, masked_live_rows)

DATA = ROOT / "data" / "SOL-ExecBench" / "benchmark"

#: The paged causal prefill shape, cut down to what the rule reads.
PAGED_CAUSAL = {
    "axes": {"num_pages": {"type": "var"}, "num_kv_indices": {"type": "var"},
             "len_indptr": {"type": "var"}, "total_q": {"type": "var"},
             "page_size": {"type": "const", "value": 1},
             "num_kv_heads": {"type": "const", "value": 4},
             "num_qo_heads": {"type": "const", "value": 32},
             "head_dim": {"type": "const", "value": 128}},
    "inputs": {
        "q": {"shape": ["total_q", "num_qo_heads", "head_dim"],
              "dtype": "bfloat16"},
        "k_cache": {"shape": ["num_pages", "page_size", "num_kv_heads",
                              "head_dim"], "dtype": "bfloat16"},
        "qo_indptr": {"shape": ["len_indptr"], "dtype": "int32"},
        "kv_indptr": {"shape": ["len_indptr"], "dtype": "int32"},
        "kv_indices": {"shape": ["num_kv_indices"], "dtype": "int32"},
    },
    "outputs": {
        "output": {"shape": ["total_q", "num_qo_heads", "head_dim"],
                   "dtype": "bfloat16"},
        "lse": {"shape": ["total_q", "num_qo_heads"], "dtype": "float32"},
    },
    "reference": '''
def run(q, k_cache, qo_indptr, kv_indptr, kv_indices):
    device = q.device
    total_q = q.shape[0]
    output = torch.zeros((total_q, 32, 128), device=device)
    lse = torch.full((total_q, 32), -float("inf"), device=device)
    q_f32 = q.to(torch.float32)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)
    for b in range(len(qo_indptr) - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        if q_start >= q_end or kv_start >= kv_end:
            continue
        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        num_kv_tokens = page_ids.shape[0]
        k_batch = k_cache_flat[page_ids]
        q_batch = q_f32[q_start:q_end]
        num_q_tokens = q_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens
        for q_idx in range(num_q_tokens):
            max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv_idx <= 0:
                continue
            output[q_start + q_idx] = k_batch[:max_kv_idx].sum(0)
    return output, lse
''',
}

PAIRING = {"axis": "total_q", "stream": "q", "stream_indptr": "qo_indptr",
           "window_axis": "num_kv_indices", "window": "kv_indices",
           "window_indptr": "kv_indptr"}

AXES = {"num_pages": 4096, "num_kv_indices": 3, "len_indptr": 2,
        "total_q": 16384, "page_size": 1, "num_kv_heads": 4,
        "num_qo_heads": 32, "head_dim": 128}


def _rewrite_reference(new_body: str) -> dict:
    return {**PAGED_CAUSAL, "reference": new_body}


def test_derives_the_whole_pairing_from_the_reference():
    """No problem name anywhere: the masked axis, the stream, and which of the
    two same-shaped indptrs cuts which extent all come out of the source."""
    assert causal_masked_axis(PAGED_CAUSAL) == PAIRING


def test_a_mask_that_does_not_skip_is_not_this_rule():
    """`FlashInfer-Bench__016/__017` mask with `masked_fill_` and produce NaN
    on an empty window. A dead row is NOT free there -- the kernel still has to
    produce the NaN -- so the live-row rule must not reach them."""
    masked_fill = _rewrite_reference('''
def run(q, k_cache, qo_indptr, kv_indptr, kv_indices):
    q_f32 = q.to(torch.float32)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)
    for b in range(len(qo_indptr) - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        num_kv_tokens = page_ids.shape[0]
        k_batch = k_cache_flat[page_ids]
        q_batch = q_f32[q_start:q_end]
        num_q_tokens = q_batch.shape[0]
        delta = num_kv_tokens - num_q_tokens
        logits = q_batch @ k_batch.T
        causal_mask = torch.arange(num_kv_tokens) >= (delta + 1)
        logits.masked_fill_(causal_mask, -float("inf"))
    return logits
''')
    assert causal_masked_axis(masked_fill) is None


def test_a_row_loop_with_no_skip_is_not_this_rule():
    """`FlashInfer-Bench__019` loops over every query row and computes it. No
    row is dead, so nothing may be discounted."""
    no_skip = _rewrite_reference(PAGED_CAUSAL["reference"].replace(
        "            if max_kv_idx <= 0:\n                continue\n", ""))
    assert causal_masked_axis(no_skip) is None


def test_a_different_causal_alignment_is_not_this_rule():
    """The closed form `min(q_len, kv_len)` is only the answer for
    `delta = kv_len - q_len`. Any other offset is a different mask, and the
    rule declines rather than applying a formula that no longer holds."""
    shifted = _rewrite_reference(PAGED_CAUSAL["reference"].replace(
        "delta = num_kv_tokens - num_q_tokens",
        "delta = num_kv_tokens - num_q_tokens * 2"))
    assert causal_masked_axis(shifted) is None


def _blob(tmp_path: Path, qo: list[int], kv: list[int],
          n_indices: int) -> dict:
    """A workload `inputs` block backed by a real safetensors file."""
    import numpy as np
    from safetensors.numpy import save_file

    path = tmp_path / "trace.safetensors"
    save_file({"qo_indptr": np.asarray(qo, dtype=np.int32),
               "kv_indptr": np.asarray(kv, dtype=np.int32),
               "kv_indices": np.zeros(n_indices, dtype=np.int32)}, str(path))
    return {name: {"type": "safetensors", "path": str(path),
                   "tensor_key": name}
            for name in ("qo_indptr", "kv_indptr", "kv_indices")}


def test_live_rows_are_summed_per_sequence(tmp_path):
    """`sum_b min(q_len, kv_len)`, not `min(total_q, num_kv_indices)`. The
    per-sequence sum is what the reference does; the axis-only form is an
    upper bound on it and would over-charge whenever there is more than one
    sequence -- which is the bound-raising direction."""
    inputs = _blob(tmp_path, [0, 100, 110], [0, 7, 40], 40)
    axes = dict(AXES, total_q=110, num_kv_indices=40, len_indptr=3)
    assert masked_live_rows(PAIRING, axes, inputs, None) == 7 + 10
    assert min(axes["total_q"], axes["num_kv_indices"]) == 40   # the loose one


def test_a_blob_that_contradicts_the_declared_axes_is_refused(tmp_path):
    """`qo_indptr[-1]` is `total_q` -- the reference asserts it. A blob that
    says otherwise is not the workload's, and the answer is no correction
    rather than a count from the wrong file."""
    inputs = _blob(tmp_path, [0, 5], [0, 3], 3)
    assert masked_live_rows(PAIRING, dict(AXES, total_q=5, num_kv_indices=3),
                            inputs, None) == 3
    assert masked_live_rows(PAIRING, dict(AXES, total_q=6, num_kv_indices=3),
                            inputs, None) is None
    assert masked_live_rows(PAIRING, dict(AXES, total_q=5, num_kv_indices=4),
                            inputs, None) is None


def test_a_workload_with_no_blob_makes_no_claim():
    """`data/` is gitignored and the FlashInfer trace blobs are fetched
    separately. Without them the tier keeps today's price."""
    inputs = {"qo_indptr": {"type": "random"}, "kv_indptr": {"type": "random"}}
    assert masked_live_rows(PAIRING, AXES, inputs, None) is None
    assert masked_live_rows(PAIRING, AXES, {}, None) is None


def test_the_stream_is_repriced_and_the_output_is_not():
    """The conservative reading, and the one three independent agents
    converged on: charge `q` at its live rows, keep charging the FULL `output`
    and `lse`. A correct kernel really does have to write `(0, -inf)` into
    every dead row. The stricter reading that discounts the output too is 125x
    here, not 2x, and is NOT taken."""
    full = gathered_traffic(PAGED_CAUSAL, AXES)      # D18 already applied
    live = gathered_traffic(PAGED_CAUSAL, AXES, live={"total_q": 3})

    row = 32 * 128 * 2                               # one bf16 q/output row
    lse = 16384 * 32 * 4
    assert full - live == (16384 - 3) * row          # exactly the dead q rows
    # the output and the lse survive the correction whole
    assert live > 16384 * row + lse
    # ~2x for the trivial reason that `q` and `output` are the same shape and
    # dtype: charging a dead `q` doubles the bound. Measured on the real
    # workload this synthesises (`__015`/`a94c44ab`): 1.9842.
    assert round(full / live, 4) == 1.9842


def test_only_the_leading_dimension_is_repriced():
    """The mask kills whole ROWS. An axis that appears somewhere other than
    first is not a row count and must keep its price."""
    inner = {**PAGED_CAUSAL,
             "inputs": {**PAGED_CAUSAL["inputs"],
                        "bias": {"shape": ["num_qo_heads", "total_q"],
                                 "dtype": "float32"}}}
    assert (gathered_traffic(inner, AXES, rewrite={}, live={"total_q": 3})
            - gathered_traffic(PAGED_CAUSAL, AXES, rewrite={},
                               live={"total_q": 3})) == 32 * 16384 * 4


def test_the_live_count_is_capped_at_the_allocation():
    """A correction may never RAISE a bound. That is the one direction this
    whole change exists to remove."""
    capped = gathered_traffic(PAGED_CAUSAL, AXES, rewrite={},
                              live={"total_q": 10 ** 9})
    assert capped == gathered_traffic(PAGED_CAUSAL, AXES, rewrite={})


def test_no_masked_axis_leaves_the_price_alone():
    plain = {
        "axes": {"n": {"type": "var"}},
        "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x + 1\n",
    }
    assert causal_masked_axis(plain) is None
    assert gathered_traffic(plain, {"n": 100}, live=None) == 100 * 4 * 2


# -- the tier end to end ----------------------------------------------------


def _tier_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A one-problem dataset the traffic-floor tier can be run over."""
    import numpy as np
    from safetensors.numpy import save_file

    data = tmp_path / "data"
    prob = data / "FlashInfer-Bench" / "014_paged_prefill_causal"
    prob.mkdir(parents=True)
    (prob / "definition.json").write_text(json.dumps(PAGED_CAUSAL))

    blob = tmp_path / "trace.safetensors"
    save_file({"qo_indptr": np.asarray([0, 100, 110], dtype=np.int32),
               "kv_indptr": np.asarray([0, 7, 40], dtype=np.int32),
               "kv_indices": np.zeros(40, dtype=np.int32)}, str(blob))
    with_blob = {name: {"type": "safetensors", "path": str(blob),
                        "tensor_key": name}
                 for name in ("qo_indptr", "kv_indptr", "kv_indices")}
    axes = {"num_pages": 4096, "num_kv_indices": 40, "len_indptr": 3,
            "total_q": 110}
    (prob / "workload.jsonl").write_text("\n".join(json.dumps(w) for w in [
        {"uuid": "with-blob", "axes": axes, "inputs": with_blob},
        {"uuid": "no-blob", "axes": axes,
         "inputs": {"qo_indptr": {"type": "random"}}},
    ]) + "\n")

    arch = tmp_path / "arch.yaml"
    arch.write_text("freq_GHz: 2.4\nDRAM_byte_per_cycle: 3333.3\n")
    t_sol = tmp_path / "t_sol.json"
    t_sol.write_text(json.dumps({"problems": {}}))
    tb = tmp_path / "tb"
    tb.mkdir()
    (tb / "FlashInfer-Bench__014_paged_prefill_causal.json").write_text(
        json.dumps({"winner_by_workload": {
            "with-blob": {"t_b_ms": 1000.0}, "no-blob": {"t_b_ms": 1000.0}}}))
    return data, arch, t_sol


def test_the_tier_reprices_the_stream_and_states_its_own_clock(tmp_path,
                                                               monkeypatch):
    """The shared field contract (D63) and the correction, together: every
    record says which clock its own `t_sol_ms` came from, and the header says
    the one clock they agree on. A cycle count on this part is only meaningful
    next to the clock it was expressed at."""
    import sol_traffic_floor

    data, arch, t_sol = _tier_tree(tmp_path)
    out = tmp_path / "t_sol_traffic.json"
    written: dict = {}

    # Mirrors `provenance.write_artifact`'s signature, `part` included: a stub
    # that quietly accepted **kwargs would go on passing while the real call
    # site drifted away from it.
    def _capture(path, task, payload, extra_provenance=None, *,
                 part=None, allow_cross_part=False):
        written.update(payload)
        written["_declared_part"] = part
        Path(path).write_text(json.dumps(payload))
        return Path(path)

    # The real one imports torch to detect the part. This tier derives on
    # `device="meta"` and the test has no business touching a GPU.
    monkeypatch.setattr(sol_traffic_floor, "write_artifact", _capture)
    monkeypatch.setattr(sys, "argv", [
        "sol_traffic_floor.py", "--data", str(data), "--arch", str(arch),
        "--t-sol", str(t_sol), "--t-b", str(tmp_path / "tb"),
        "--out", str(out)])
    assert sol_traffic_floor.main() == 0

    assert written["f_ref_mhz"] == 2400.0
    records = (written["problems"]
               ["FlashInfer-Bench__014_paged_prefill_causal"]["workloads"])
    assert {r["f_ref_mhz"] for r in records.values()} == {written["f_ref_mhz"]}

    live, dark = records["with-blob"], records["no-blob"]
    assert live["masked_axis"] == "total_q"
    assert live["masked_rows"] == 7 + 10
    assert live["memory_bytes"] < live["gathered_bytes"] < live["allocation_bytes"]
    # No blob, no claim: the workload keeps the price it had before this rule.
    assert dark["masked_rows"] is None
    assert dark["memory_bytes"] == dark["gathered_bytes"]

    stats = written["stats"]
    assert stats["problems_with_masked_stream"] == 1
    assert stats["workloads_repriced_from_stream_to_live_rows"] == 1
    assert stats["workloads_masked_rows_unresolved"] == 1

    # No `--part`, so nothing is DECLARED and the stamper is left to detect it.
    # An undeclared artifact is a legible absence; a declared wrong one is not.
    assert written["_declared_part"] is None


def test_the_tier_declares_the_part_it_was_asked_for(tmp_path, monkeypatch):
    """`--part` reaches the stamper, so a wrong one is refused there.

    `--arch` defaults to MI350X.yaml. Without a declaration, an MI355X node that
    forgets the flag writes MI350X bandwidth into a file that every downstream
    inference path -- device names, hostname -- reads as MI355X, and no check in
    the tree can see it. The declaration is what gives
    `provenance.stamp()` something to contradict.
    """
    import sol_traffic_floor

    data, arch, t_sol = _tier_tree(tmp_path)
    seen: dict = {}

    def _capture(path, task, payload, extra_provenance=None, *,
                 part=None, allow_cross_part=False):
        seen["part"] = part
        Path(path).write_text(json.dumps(payload))
        return Path(path)

    monkeypatch.setattr(sol_traffic_floor, "write_artifact", _capture)
    monkeypatch.setattr(sys, "argv", [
        "sol_traffic_floor.py", "--data", str(data), "--arch", str(arch),
        "--t-sol", str(t_sol), "--t-b", str(tmp_path / "tb"),
        "--out", str(tmp_path / "t_sol_traffic.json"), "--part", "MI355X"])
    assert sol_traffic_floor.main() == 0
    assert seen["part"] == "MI355X"


# -- the dataset itself -----------------------------------------------------
# Anchored on the measured numbers, so a rewrite of the derivation that still
# passes the synthetic cases cannot quietly change the answer on the real
# problems. Skipped where `data/` has not been materialised.

_LIVE = {
    ("014_gqa_paged_prefill_causal_h32_kv4_d128_ps1",
     "d14e12cc-4fd1-43d2-a156-4e784c4f252d"): (15783, 25),
    ("014_gqa_paged_prefill_causal_h32_kv4_d128_ps1",
     "3e553162-e3e4-446c-b72d-546bbf07c495"): (16384, 2),
    ("015_gqa_paged_prefill_causal_h32_kv8_d128_ps1",
     "a94c44ab-5899-419f-9c79-9898fda0e173"): (16384, 3),
    ("015_gqa_paged_prefill_causal_h32_kv8_d128_ps1",
     "3b672ff1-11ba-4d30-bd7e-303472aecf0b"): (10447, 1),
}


@pytest.mark.skipif(not (DATA / "FlashInfer-Bench").is_dir(),
                    reason="dataset not materialised")
@pytest.mark.parametrize("problem,uuid", sorted(_LIVE))
def test_measured_live_rows_on_the_falsifying_workloads(problem, uuid):
    prob = DATA / "FlashInfer-Bench" / problem
    definition = json.loads((prob / "definition.json").read_text())
    spec = causal_masked_axis(definition)
    assert spec == PAIRING
    for line in (prob / "workload.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        w = json.loads(line)
        if w["uuid"] != uuid:
            continue
        total_q, live = _LIVE[(problem, uuid)]
        assert w["axes"]["total_q"] == total_q
        assert masked_live_rows(spec, w["axes"], w["inputs"], ROOT) == live
        return
    pytest.fail(f"{uuid} not in {problem}/workload.jsonl")


@pytest.mark.skipif(not (DATA / "FlashInfer-Bench").is_dir(),
                    reason="dataset not materialised")
def test_the_rule_fires_on_two_problems_and_no_others():
    """The gate is a property of the workload, not an allowlist -- but over
    the whole 235 it must still land on exactly the two problems whose
    references skip. `__016`, `__017` and `__019` are the near misses."""
    fired = set()
    for cat in sorted(DATA.iterdir()):
        if not cat.is_dir():
            continue
        for prob in sorted(cat.iterdir()):
            defn = prob / "definition.json"
            if not defn.exists():
                continue
            if causal_masked_axis(json.loads(defn.read_text())):
                fired.add(f"{cat.name}__{prob.name}")
    assert fired == {
        "FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1",
        "FlashInfer-Bench__015_gqa_paged_prefill_causal_h32_kv8_d128_ps1"}
