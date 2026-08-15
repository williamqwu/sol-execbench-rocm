# SPDX-License-Identifier: Apache-2.0
"""D18: the declared-traffic tier must price a gather, not an allocation.

`FlashInfer-Bench__015`'s worst workload declares `num_pages = 552310` and
names two of them in `kv_indices`. Priced at the allocation the bound is
0.282821 ms -- 43x above a real Triton kernel's measured 0.00664 ms, so it is
not a bound at all. Priced at the two pages it is 3.61e-5 ms.

The pairing must come out of the problem definition. These tests pin the three
places the derivation can go wrong: the gather must be found through aliases
and casts, a SLICE whose bounds come from an index vector must not be mistaken
for one, and an OUTPUT dimensioned by the allocation must keep its full price
because a scatter still has to write every row.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from sol_gathered_traffic import gathered_axes, gathered_traffic  # noqa: E402

PAGED = {
    "axes": {"num_pages": {"type": "var"}, "num_kv_indices": {"type": "var"},
             "len_indptr": {"type": "var"}, "total_q": {"type": "var"},
             "page_size": {"type": "const", "value": 1},
             "num_kv_heads": {"type": "const", "value": 8},
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
    "outputs": {"output": {"shape": ["total_q", "num_qo_heads", "head_dim"],
                           "dtype": "bfloat16"}},
    "reference": '''
def run(q, k_cache, qo_indptr, kv_indptr, kv_indices):
    device = q.device
    total_q = q.shape[0]
    output = torch.zeros((total_q, 32, 128), device=device)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)
    q_f32 = q.to(torch.float32)
    for b in range(len(qo_indptr) - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        k_batch = k_cache_flat[page_ids]
        q_batch = q_f32[q_start:q_end]
        for i in range(q_batch.shape[0]):
            output[q_start + i] = k_batch.sum(0)
    return output
''',
}

AXES = {"num_pages": 552310, "num_kv_indices": 2, "len_indptr": 2,
        "total_q": 17, "page_size": 1, "num_kv_heads": 8,
        "num_qo_heads": 32, "head_dim": 128}


def test_pairs_the_allocation_axis_with_the_index_vector():
    assert gathered_axes(PAGED) == {"num_pages": "num_kv_indices"}


def test_a_slice_is_not_a_gather():
    """`q_f32[q_start:q_end]` and `output[q_start + i]` both take their bounds
    from `qo_indptr`, an int32 vector of length `len_indptr`. Reading either as
    a gather would price `total_q` at `len_indptr` -- 2 rows of q instead of
    17 -- which is the correction pointed at the wrong tensor."""
    assert "total_q" not in gathered_axes(PAGED)


def test_prices_the_cache_at_the_pages_named():
    got = gathered_traffic(PAGED, AXES)
    q_out = 2 * 17 * 32 * 128 * 2          # q read, output written
    cache = 2 * 1 * 8 * 128 * 2            # TWO pages, not 552310
    indptrs = 2 * 2 * 4 + 2 * 4            # qo_indptr, kv_indptr, kv_indices
    assert got == q_out + cache + indptrs
    # and the allocation price it replaces, for the size of the defect
    alloc = gathered_traffic(PAGED, AXES, rewrite={})
    assert alloc - got == (552310 - 2) * 1 * 8 * 128 * 2
    assert alloc / got > 4000                      # three orders of magnitude


def test_an_output_dimensioned_by_the_allocation_keeps_its_full_price():
    """A gather is a claim about READING. The matching scatter writes into a
    tensor that still exists in full -- every row of it, zeros included."""
    scatter = {
        "axes": {"batch_seq_len": {"type": "var"},
                 "num_tokens": {"type": "var"},
                 "hidden_dim": {"type": "const", "value": 4}},
        "inputs": {
            "grad_output": {"shape": ["batch_seq_len", "hidden_dim"],
                            "dtype": "float32"},
            "token_indices": {"shape": ["num_tokens"], "dtype": "int64"}},
        "outputs": {"grad_hidden_states": {
            "shape": ["batch_seq_len", "hidden_dim"], "dtype": "float32"}},
        "reference": '''
def run(grad_output, token_indices):
    gathered = grad_output[token_indices]
    out = torch.zeros_like(grad_output)
    out[token_indices] = gathered
    return out
''',
    }
    axes = {"batch_seq_len": 1000, "num_tokens": 10, "hidden_dim": 4}
    assert gathered_axes(scatter) == {"batch_seq_len": "num_tokens"}
    got = gathered_traffic(scatter, axes)
    read = 10 * 4 * 4 + 10 * 8            # 10 rows gathered + the index vector
    written = 1000 * 4 * 4                # the whole output, scatter or not
    assert got == read + written


def test_the_gathered_count_is_capped_at_the_allocation():
    """An index vector may name a slot twice. `FlashInfer-Bench__015` /
    `75ab4c21` names 28 with `num_pages = 2` -- one distinct page, repeated --
    and no kernel reads more distinct rows than exist. Uncapped, the
    "correction" RAISED that bound, which is the direction it exists to
    remove."""
    axes = dict(AXES, num_pages=2, num_kv_indices=28)
    capped = gathered_traffic(PAGED, axes)
    allocation = gathered_traffic(PAGED, axes, rewrite={})
    assert capped == allocation                # 2 pages either way
    assert capped < gathered_traffic(PAGED, dict(axes, num_pages=28))


def test_no_index_vector_means_no_rewrite():
    plain = {
        "axes": {"n": {"type": "var"}},
        "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
        "reference": "def run(x):\n    return x + 1\n",
    }
    assert gathered_axes(plain) == {}
    assert gathered_traffic(plain, {"n": 100}) == 100 * 4 * 2


def test_unresolved_gather_axis_makes_no_claim():
    """None, not a partial count: half the correction is neither number."""
    assert gathered_traffic(PAGED, {k: v for k, v in AXES.items()
                                    if k != "num_kv_indices"}) is None
