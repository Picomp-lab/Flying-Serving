# SPDX-License-Identifier: Apache-2.0
"""Token accounting across a DP<->TP switch preemption.

When the merged-engine count changes, the KV block layout changes with it, so
every in-flight request has its blocks freed and is re-prefilled. The tokens it
already generated are folded into its prompt by
`Request.reset_output_token_ids` so they are not emitted twice -- which means
they leave `_output_token_ids`, which is what the generation budget used to be
measured against.
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus

MAX_MODEL_LEN = 4096


def make_request(max_tokens: int = 10, min_tokens: int = 0) -> Request:
    return Request(
        request_id="r0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            ignore_eos=True,
        ),
        pooling_params=None,
        eos_token_id=None,
    )


def test_fold_preserves_generated_tokens_in_prompt():
    """The folded tokens must actually land in the prompt.

    `reset_output_token_ids(req._output_token_ids)` passes the request's own
    list, so clearing before copying makes the extend a no-op -- the request
    then re-prefills from the original prompt while `_all_token_ids` still
    counts the generated tokens, and the worker is asked to compute over
    positions it was never given.
    """
    req = make_request()
    req.append_output_token_ids([11, 12, 13])
    assert req.num_tokens == 6

    req.reset_output_token_ids(req._output_token_ids)

    assert list(req.prompt_token_ids) == [1, 2, 3, 11, 12, 13]
    assert req.num_prompt_tokens == 6
    # The prompt now spans everything the request holds; a re-prefill of
    # `num_tokens` positions is fully backed by real token ids.
    assert req.num_prompt_tokens == req.num_tokens
    assert req.num_output_tokens == 0


def test_max_tokens_is_not_reset_by_the_switch():
    """max_tokens counts the whole lifetime, not the post-switch segment."""
    req = make_request(max_tokens=10)
    req.append_output_token_ids(list(range(100, 106)))  # 6 generated
    assert not check_stop(req, MAX_MODEL_LEN)

    # Engines switch mode: preempt and fold.
    req.reset_output_token_ids(req._output_token_ids)
    req.status = RequestStatus.WAITING
    req.num_computed_tokens = 0

    assert req.num_output_tokens == 0  # physical list is empty
    assert req.num_output_tokens_total == 6  # budget remembers

    # 3 more tokens: 9 of 10, still running.
    req.append_output_token_ids(list(range(200, 203)))
    assert req.num_output_tokens_total == 9
    assert not check_stop(req, MAX_MODEL_LEN)

    # The 10th token ends it -- not the 10th *after the switch*.
    req.append_output_token_ids(203)
    assert req.num_output_tokens_total == 10
    assert check_stop(req, MAX_MODEL_LEN)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_repeated_switches_accumulate():
    """Several switches during one request must not each reset the budget."""
    req = make_request(max_tokens=9)
    for _ in range(3):
        req.append_output_token_ids([1, 2, 3])
        req.reset_output_token_ids(req._output_token_ids)
    assert req.num_output_tokens_total == 9
    assert check_stop(req, MAX_MODEL_LEN)


def test_min_tokens_also_counts_folded_tokens():
    """min_tokens must not demand a fresh quota after a switch either."""
    req = make_request(max_tokens=100, min_tokens=4)
    req.append_output_token_ids([1, 2, 3, 4])
    req.reset_output_token_ids(req._output_token_ids)
    # eos would otherwise be suppressed because the physical list is empty.
    req.sampling_params.ignore_eos = False
    req.eos_token_id = 42
    req.append_output_token_ids(42)
    assert check_stop(req, MAX_MODEL_LEN)
    assert req.status == RequestStatus.FINISHED_STOPPED


def test_untouched_requests_are_unaffected():
    """A request that never crosses a switch behaves exactly as before."""
    req = make_request(max_tokens=3)
    req.append_output_token_ids([1, 2])
    assert req.num_output_tokens_total == req.num_output_tokens == 2
    assert not check_stop(req, MAX_MODEL_LEN)
    req.append_output_token_ids(3)
    assert check_stop(req, MAX_MODEL_LEN)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
