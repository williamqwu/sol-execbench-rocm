# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure scoring logic. No GPU, no agents, no filesystem sweep.

These cover the cases where returning a plausible number instead of None would
be actively harmful, because a bad score is indistinguishable from a good one
once it is in a table.
"""

from __future__ import annotations

import textwrap

import pytest

from solexbench_agents.scoring import (
    ScoreBasis,
    compare_digests,
    headroom_fraction,
    reference_copy_verdict,
    resolve_basis,
    sol_score,
    tree_digest,
)


class TestSolScore:
    def test_one_at_the_sol_bound(self):
        assert sol_score(t_k_ms=1.0, t_sol_ms=1.0, t_b_ms=4.0) == pytest.approx(1.0)

    def test_half_at_the_anchor(self):
        """S = 0.5 when the kernel matches T_b. This is the scale's definition."""
        assert sol_score(t_k_ms=4.0, t_sol_ms=1.0, t_b_ms=4.0) == pytest.approx(0.5)

    def test_monotonically_decreasing_in_t_k(self):
        scores = [sol_score(t, 1.0, 4.0) for t in (1.0, 2.0, 4.0, 8.0, 16.0)]
        assert scores == sorted(scores, reverse=True)

    def test_slower_than_anchor_scores_below_half(self):
        assert sol_score(10.0, 1.0, 4.0) < 0.5

    def test_none_when_anchor_at_or_below_bound(self):
        """A broken bound must not yield a number.

        t_b <= t_sol makes the denominator zero or negative. Returning a value
        would hide a bound that cannot be right, since an optimized-PyTorch
        anchor cannot reach the Speed-of-Light limit.
        """
        assert sol_score(2.0, 4.0, 4.0) is None
        assert sol_score(2.0, 5.0, 4.0) is None

    def test_none_when_any_input_missing(self):
        assert sol_score(None, 1.0, 4.0) is None
        assert sol_score(2.0, None, 4.0) is None
        assert sol_score(2.0, 1.0, None) is None

    def test_exceeds_one_when_kernel_beats_the_bound(self):
        """Not clamped. S > 1 is the visible symptom of a too-loose bound."""
        assert sol_score(0.5, 1.0, 4.0) > 1.0


class TestHeadroomFraction:
    def test_zero_at_the_reference(self):
        assert headroom_fraction(4.0, 4.0, 1.0) == pytest.approx(0.0)

    def test_one_at_the_bound(self):
        assert headroom_fraction(1.0, 4.0, 1.0) == pytest.approx(1.0)

    def test_negative_when_slower_than_the_reference(self):
        assert headroom_fraction(5.0, 4.0, 1.0) < 0

    def test_none_when_reference_is_at_or_below_the_bound(self):
        assert headroom_fraction(2.0, 1.0, 1.0) is None
        assert headroom_fraction(2.0, 1.0, 4.0) is None


class TestResolveBasis:
    def test_incorrect_is_correctness_only_even_with_every_bound(self):
        assert resolve_basis(correct=False, t_k_ms=1.0, t_ref_ms=2.0,
                             t_sol_ms=0.5, t_b_ms=2.0) is ScoreBasis.CORRECTNESS_ONLY

    def test_full_score_needs_both_bounds(self):
        assert resolve_basis(correct=True, t_k_ms=1.0, t_ref_ms=2.0,
                             t_sol_ms=0.5, t_b_ms=2.0) is ScoreBasis.SOL_SCORE_V1

    def test_headroom_without_t_b(self):
        assert resolve_basis(correct=True, t_k_ms=1.0, t_ref_ms=2.0,
                             t_sol_ms=0.5, t_b_ms=None) is ScoreBasis.SOL_HEADROOM

    def test_speedup_without_t_sol(self):
        assert resolve_basis(correct=True, t_k_ms=1.0, t_ref_ms=2.0,
                             t_sol_ms=None, t_b_ms=None) \
            is ScoreBasis.SPEEDUP_VS_REFERENCE

    def test_no_timing_is_correctness_only(self):
        assert resolve_basis(correct=True, t_k_ms=None, t_ref_ms=None,
                             t_sol_ms=None, t_b_ms=None) \
            is ScoreBasis.CORRECTNESS_ONLY


REFERENCE = textwrap.dedent("""
    import torch

    @torch.no_grad()
    def run(x, w, eps):
        # normalize
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + eps) * w.float()).to(x.dtype)
""")


class TestReferenceCopyVerdict:
    def test_exact_copy_detected(self):
        v = reference_copy_verdict([{"content": REFERENCE}], REFERENCE)
        assert v.kind == "exact"
        assert v.is_copy

    def test_comments_and_formatting_do_not_hide_a_copy(self):
        disguised = REFERENCE.replace("# normalize", "# compute the variance") \
                             .replace("\n\n", "\n\n\n")
        v = reference_copy_verdict([{"content": disguised}], REFERENCE)
        assert v.kind == "exact", "AST comparison should ignore comments"

    def test_docstring_addition_does_not_hide_a_copy(self):
        disguised = REFERENCE.replace(
            "def run(x, w, eps):", 'def run(x, w, eps):\n    """RMS norm."""'
        )
        v = reference_copy_verdict([{"content": disguised}], REFERENCE)
        assert v.kind == "exact"

    def test_syntactically_broken_source_does_not_crash(self):
        """A source that does not parse still gets a verdict.

        Reached via the text-similarity fallback, which is also the path a HIP or
        CUDA source takes. It must never raise: a submission that fails to parse
        is a thing to record, not a reason the scorer dies.
        """
        broken = REFERENCE.replace("def run(x, w, eps):",
                                   "def run(x, w, eps):\n        oops")
        v = reference_copy_verdict([{"content": broken}], REFERENCE)
        assert v.kind in ("distinct", "near")

    def test_genuine_triton_kernel_is_distinct(self):
        kernel = textwrap.dedent("""
            import torch, triton, triton.language as tl

            @triton.jit
            def _rms(x_ptr, w_ptr, o_ptr, eps, n, BLOCK: tl.constexpr):
                row = tl.program_id(0)
                offs = tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + row * n + offs, mask=mask, other=0.0)
                acc = tl.sum(x * x, axis=0) / n
                w = tl.load(w_ptr + offs, mask=mask, other=1.0)
                tl.store(o_ptr + row * n + offs, x * tl.rsqrt(acc + eps) * w, mask=mask)

            def run(x, w, eps):
                out = torch.empty_like(x)
                n = x.shape[-1]
                _rms[(x.numel() // n,)](x, w, out, eps, n, BLOCK=triton.next_power_of_2(n))
                return out
        """)
        v = reference_copy_verdict([{"content": kernel}], REFERENCE)
        assert v.kind == "distinct"
        assert not v.is_copy

    def test_unparseable_source_falls_back_to_text_similarity(self):
        hip = "__global__ void k(float* x) { x[threadIdx.x] *= 2.0f; }"
        v = reference_copy_verdict([{"content": hip}], REFERENCE)
        assert v.kind == "distinct"

    def test_picks_the_closest_of_several_sources(self):
        v = reference_copy_verdict(
            [{"content": "import torch\n"}, {"content": REFERENCE}], REFERENCE
        )
        assert v.kind == "exact"

    def test_empty_sources_are_distinct_not_a_crash(self):
        assert reference_copy_verdict([], REFERENCE).kind == "distinct"
        assert reference_copy_verdict([{"content": ""}], REFERENCE).kind == "distinct"


class TestTreeDigest:
    def test_stable_across_calls(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
        first = tree_digest(tmp_path, ["pkg"])
        second = tree_digest(tmp_path, ["pkg"])
        assert first["sha256"] == second["sha256"]
        assert first["n_files"] == 1

    def test_content_change_is_detected(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        f = tmp_path / "pkg" / "a.py"
        f.write_text("x = 1\n")
        before = tree_digest(tmp_path, ["pkg"])
        f.write_text("x = 2\n")
        after = tree_digest(tmp_path, ["pkg"])
        cmp = compare_digests(before, after)
        assert cmp["comparable"] and not cmp["match"]
        assert cmp["changed"] == ["pkg/a.py"]

    def test_added_and_removed_files_are_reported(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
        before = tree_digest(tmp_path, ["pkg"])
        (tmp_path / "pkg" / "a.py").unlink()
        (tmp_path / "pkg" / "b.py").write_text("y = 1\n")
        cmp = compare_digests(before, tree_digest(tmp_path, ["pkg"]))
        assert cmp["added"] == ["pkg/b.py"]
        assert cmp["removed"] == ["pkg/a.py"]

    def test_missing_baseline_is_not_silently_a_pass(self, tmp_path):
        cmp = compare_digests(None, tree_digest(tmp_path, []))
        assert cmp["comparable"] is False
        assert "cannot verify" in cmp["note"]


class TestGpuPool:
    def test_authoritative_gpu_refused(self):
        from solexbench_agents.gpu_pool import AUTHORITATIVE_GPU, GpuPool

        with pytest.raises(ValueError, match="reserved for authoritative timing"):
            GpuPool([AUTHORITATIVE_GPU, AUTHORITATIVE_GPU + 1])

    def test_lease_returns_the_gpu(self):
        from solexbench_agents.gpu_pool import default_agent_gpus, GpuPool

        two = default_agent_gpus(8)[:2]
        pool = GpuPool(two)
        with pool.lease() as g:
            assert g in two
        with pool.lease() as a, pool.lease() as b:
            assert {a, b} == set(two)

    def test_pool_bounds_concurrency(self):
        """The D11 property: a third holder cannot exist while two are out."""
        import queue

        from solexbench_agents.gpu_pool import default_agent_gpus, GpuPool

        pool = GpuPool(default_agent_gpus(8)[:2])
        with pool.lease(), pool.lease():
            with pytest.raises(queue.Empty):
                with pool.lease(timeout=0.05):
                    pass

    def test_empty_pool_refused(self):
        from solexbench_agents.gpu_pool import GpuPool

        with pytest.raises(ValueError):
            GpuPool([])

    def test_default_pool_excludes_authoritative(self):
        from solexbench_agents.gpu_pool import AUTHORITATIVE_GPU, default_agent_gpus

        gpus = default_agent_gpus(8)
        assert AUTHORITATIVE_GPU not in gpus
        assert len(gpus) == 7
