# SPDX-License-Identifier: Apache-2.0
"""Subset merging: which engines agree to form one dynamic-TP group.

With more than one mergeable subset, entering TP mode stops being a
whole-deployment decision. These cover the rule that replaces it
(`dtp_switch_agreed`) and the subset enumeration it is fed from.
"""

import dataclasses

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.v1.engine.core import dtp_switch_agreed

DP = 4


def row(want: bool, switch_id: int, group: tuple[int, ...]) -> list[int]:
    mask = [1 if e in group else 0 for e in range(DP)]
    return [0, int(want), switch_id, *mask]


def matrix(*rows: list[int]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64)


# --------------------------------------------------------------------------- #
#  Merge-set enumeration
# --------------------------------------------------------------------------- #


def test_merge_sets_tile_the_deployment_at_every_width():
    """Every width must cover all engines: that is what DP<->TP buys over
    shrinking a TP group, which leaves the released GPUs idle."""
    for dp in (2, 4, 8):
        sets = ParallelConfig.dtp_merge_sets(dp)
        by_width: dict[int, list[tuple[int, ...]]] = {}
        for grp in sets:
            by_width.setdefault(len(grp), []).append(grp)
        for width, groups in by_width.items():
            covered = sorted(e for g in groups for e in g)
            assert covered == list(range(dp)), (
                f"dp={dp} width={width} leaves engines idle: {groups}")
            assert len(covered) == len(set(covered)), "engine in two groups"


def test_merge_sets_include_the_full_merge():
    for dp in (2, 4, 6, 8):
        assert tuple(range(dp)) in ParallelConfig.dtp_merge_sets(dp)


def test_merge_sets_are_linear_not_combinatorial():
    # `itertools.combinations` over dp=8 is 246 subsets of size >= 2; one
    # communicator each is the reason the full enumeration was never used.
    assert len(ParallelConfig.dtp_merge_sets(8)) == 7


def test_single_engine_has_nothing_to_merge():
    assert ParallelConfig.dtp_merge_sets(1) == []


def test_merge_sets_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_DTP_MERGE_SETS", "0,1;2,3")
    assert ParallelConfig.dtp_merge_sets(4) == [(0, 1), (2, 3)]
    monkeypatch.setenv("VLLM_DTP_MERGE_SETS", "0,9")
    with pytest.raises(ValueError):
        ParallelConfig.dtp_merge_sets(4)


# --------------------------------------------------------------------------- #
#  Group agreement
# --------------------------------------------------------------------------- #


def test_two_groups_merge_independently():
    """The case the old all-engines rule could not express."""
    t = matrix(
        row(True, 111, (0, 1)),
        row(True, 111, (0, 1)),
        row(True, 222, (2, 3)),
        row(True, 222, (2, 3)),
    )
    assert all(dtp_switch_agreed(t, r, DP) for r in range(DP))


def test_one_group_merges_while_the_rest_stay_in_dp():
    t = matrix(
        row(True, 111, (0, 1)),
        row(True, 111, (0, 1)),
        row(False, 0, ()),
        row(False, 0, ()),
    )
    assert dtp_switch_agreed(t, 0, DP)
    assert dtp_switch_agreed(t, 1, DP)
    # Engines 2 and 3 propose nothing and must not be dragged into TP mode.
    assert not dtp_switch_agreed(t, 2, DP)
    assert not dtp_switch_agreed(t, 3, DP)


def test_the_old_whole_deployment_merge_still_agrees():
    full = (0, 1, 2, 3)
    t = matrix(*[row(True, 7, full) for _ in range(DP)])
    assert all(dtp_switch_agreed(t, r, DP) for r in range(DP))


def test_a_member_that_has_not_voted_blocks_its_group_only():
    """Engine 1 has not seen the switch request yet: its group waits, the
    other group does not."""
    t = matrix(
        row(True, 111, (0, 1)),
        row(False, 0, ()),
        row(True, 222, (2, 3)),
        row(True, 222, (2, 3)),
    )
    assert not dtp_switch_agreed(t, 0, DP)
    assert dtp_switch_agreed(t, 2, DP)
    assert dtp_switch_agreed(t, 3, DP)


def test_members_naming_different_switch_requests_do_not_merge():
    """Two switch requests in flight at once must not be conflated: the
    engines would enter TP mode expecting different first batches."""
    t = matrix(
        row(True, 111, (0, 1)),
        row(True, 999, (0, 1)),
        row(False, 0, ()),
        row(False, 0, ()),
    )
    assert not dtp_switch_agreed(t, 0, DP)
    assert not dtp_switch_agreed(t, 1, DP)


def test_disagreeing_proposals_do_not_merge():
    """Engine 0 wants (0,1); engine 1 wants the full merge. Neither may
    proceed -- a merge where the members disagree on the member list is
    exactly the shape mismatch that hangs the merged all-reduce."""
    t = matrix(
        row(True, 111, (0, 1)),
        row(True, 111, (0, 1, 2, 3)),
        row(True, 111, (0, 1, 2, 3)),
        row(True, 111, (0, 1, 2, 3)),
    )
    assert not dtp_switch_agreed(t, 0, DP)
    assert not dtp_switch_agreed(t, 1, DP)


def test_a_group_of_one_is_not_a_merge():
    t = matrix(
        row(True, 111, (0,)),
        row(False, 0, ()),
        row(False, 0, ()),
        row(False, 0, ()),
    )
    assert not dtp_switch_agreed(t, 0, DP)


def test_verdict_is_unanimous_within_a_group():
    """Every member computes the same answer from the same matrix -- that is
    what makes the flip land on the same step without a second collective."""
    for group in ((0, 1), (2, 3), (0, 1, 2, 3)):
        t = matrix(*[
            row(True, 5, group) if e in group else row(False, 0, ())
            for e in range(DP)
        ])
        verdicts = {dtp_switch_agreed(t, e, DP) for e in group}
        assert verdicts == {True}




# --------------------------------------------------------------------------- #
#  The merged KV layout
# --------------------------------------------------------------------------- #
#
# A merged rank owns 1/d of the heads, so the same KV bytes hold d times as many
# token slots: block_size scales up, num_kv_heads scales down. Four consumers
# have to agree on that reinterpretation -- the cache spec, the attention
# metadata builder, the block table (which turns a block id into a slot index)
# and the model's own view of the cache tensor. Any one of them left at a
# different width and the slot a token is *written* to stops being the slot
# attention *reads*, with no error and no hang: the rank returns whatever lives
# at the address it read.


@dataclasses.dataclass
class _Spec:
    block_size: int
    num_kv_heads: int


def fake_runner(block_size: int = 16, num_kv_heads: int = 8,
                num_heads_q: int = 32):
    """A GPUModelRunner reduced to the attributes the layout code touches."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    builder = SimpleNamespace(block_size=block_size,
                              num_heads_kv=num_kv_heads,
                              num_heads_q=num_heads_q)
    block_table = SimpleNamespace(block_size=block_size)
    group = SimpleNamespace(kv_cache_spec=_Spec(block_size, num_kv_heads))
    r = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group],
                                        change_status_for_dtp=False),
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[block_table])),
        attn_groups=[[SimpleNamespace(get_metadata_builder=lambda: builder)]],
        model=SimpleNamespace(model=SimpleNamespace(
            long_request_engine_ids=(0, 1))),
        original_status={},
        dtp_applied_ids=(),
        dtp_context_switch_status=False,
        long_request_engine_ids=[0, 1],
    )
    for name in ("dtp_apply_layout", "dtp_undo_layout",
                 "_dtp_metadata_builder"):
        setattr(r, name, getattr(GPUModelRunner, name).__get__(r))
    return r, group, builder, block_table


def layout(runner):
    _, group, builder, block_table = runner
    return (group.kv_cache_spec.block_size, group.kv_cache_spec.num_kv_heads,
            builder.block_size, builder.num_heads_kv, builder.num_heads_q,
            block_table.block_size)


def test_merging_scales_every_consumer_together():
    r = fake_runner()
    r[0].dtp_apply_layout((0, 1, 2, 3))
    # block_size x4, kv heads /4, and the query head count the FA3 ahead-of-time
    # tile schedule is built from /4 as well.
    assert layout(r) == (64, 2, 64, 2, 8, 64)


def test_layout_round_trip_is_exact():
    r = fake_runner()
    before = layout(r)
    r[0].dtp_apply_layout((2, 3))
    assert layout(r) != before
    r[0].dtp_undo_layout()
    assert layout(r) == before
    assert r[0].dtp_applied_ids == ()
    assert not r[0].kv_cache_config.change_status_for_dtp


def test_leaving_tp_restores_the_layout_with_nothing_scheduled():
    """The bug this guards.

    The undo used to live in `execute_model`'s `dtp_context`, which is reached
    only on a step that schedules tokens -- and the step that leaves TP mode
    often schedules none, because the hard-preempt exit frees the KV of
    everything in flight and re-queues it. An engine that skipped it kept the
    merged block_size into the next episode.
    """
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker

    r = fake_runner()
    before = layout(r)
    worker = SimpleNamespace(model_runner=r[0], rank=3)

    Worker.worker_set_dtp_group_state(worker, True, [2, 3])
    assert layout(r) == (32, 4, 32, 4, 16, 32)
    # No forward, no scheduled tokens -- just the state flip.
    Worker.worker_set_dtp_group_state(worker, False, None)
    assert layout(r) == before


def test_a_second_episode_at_a_new_width_starts_from_the_unmerged_layout():
    """1 -> 2 -> 4 must land on exactly the layout of 1 -> 4.

    Applying width 4 on top of a stale width-2 layout is what produced a rank
    whose attention output was exactly zero: it wrote token t of block b at slot
    b*32+t while the cache view and the kernel read it at b*64+t.
    """
    direct = fake_runner()
    direct[0].dtp_apply_layout((0, 1, 2, 3))

    staged = fake_runner()
    staged[0].dtp_apply_layout((2, 3))
    staged[0].dtp_undo_layout()
    staged[0].dtp_apply_layout((0, 1, 2, 3))

    assert layout(staged) == layout(direct)


def test_re_entry_without_an_intervening_undo_does_not_compound():
    """Belt and braces: even if the exit is missed entirely, the new width is
    applied to the unmerged layout rather than on top of the old one."""
    direct = fake_runner()
    direct[0].dtp_apply_layout((0, 1, 2, 3))

    stale = fake_runner()
    stale[0].dtp_apply_layout((2, 3))
    stale[0].dtp_apply_layout((0, 1, 2, 3))   # no undo in between

    assert layout(stale) == layout(direct)


def test_undo_is_idempotent_and_safe_before_any_merge():
    r = fake_runner()
    before = layout(r)
    r[0].dtp_undo_layout()
    r[0].dtp_undo_layout()
    assert layout(r) == before


def test_undo_does_not_trust_the_pushed_engine_set():
    """`worker_set_dtp_group_state` overwrites `long_request_engine_ids` with
    the set being *entered*. An undo that divided by that instead of restoring
    saved values would unmerge a width-2 layout by four."""
    r = fake_runner()
    before = layout(r)
    r[0].dtp_apply_layout((2, 3))
    r[0].long_request_engine_ids = (0, 1, 2, 3)   # next episode, pushed early
    r[0].dtp_undo_layout()
    assert layout(r) == before


# --------------------------------------------------------------------------- #
#  The merged engine set reaches the workers at the transition
# --------------------------------------------------------------------------- #


def test_worker_learns_the_merged_engine_set_at_the_transition():
    """A dummy batch runs the model with the DTP path active but never enters
    `execute_model`'s `dtp_context`, so it uses whatever engine set the model
    was last told about. The model's own default is `(0, 1)` -- correct only at
    data_parallel_size 2. Pushing the set at the transition is what keeps an
    engine of group (2,3) from all-reducing on `_DTP[(0,1)]`.
    """
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker

    r = fake_runner()
    worker = SimpleNamespace(model_runner=r[0], rank=3)

    Worker.worker_set_dtp_group_state(worker, True, [3, 2])

    assert r[0].long_request_engine_ids == (2, 3)          # sorted
    assert r[0].model.model.long_request_engine_ids == (2, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
