import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """A logit processor that controls the length of thinking."""

    THINKING_START_TOKEN_ID: int
    THINKING_END_TOKEN_ID: int
    NEW_LINE_TOKEN_ID: int

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            thinking_budget: int | None = param_dict.get("thinking_budget")

            # Skip if thinking_budget is unset, or not an integer, or negative
            if (
                thinking_budget is None
                or not isinstance(thinking_budget, int)
                or thinking_budget < 0
            ):
                continue
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if (
                self.THINKING_START_TOKEN_ID not in cur_ids
                or self.THINKING_END_TOKEN_ID in cur_ids
            ):
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(self.THINKING_START_TOKEN_ID)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if not req.output_ids or req.output_ids[-1] != self.NEW_LINE_TOKEN_ID:
                logits[i, :] = -float("inf")
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, self.THINKING_END_TOKEN_ID] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models."""

    THINKING_START_TOKEN_ID: int = 151350
    THINKING_END_TOKEN_ID: int = 151351
    NEW_LINE_TOKEN_ID: int = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for Qwen3 models."""

    THINKING_START_TOKEN_ID: int = 151667
    THINKING_END_TOKEN_ID: int = 151668
    NEW_LINE_TOKEN_ID: int = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """A logit processor that controls the length of thinking for DeepSeek-R1 models."""

    THINKING_START_TOKEN_ID: int = 128798
    THINKING_END_TOKEN_ID: int = 128799
    NEW_LINE_TOKEN_ID: int = 201


class KimiK25ReasoningEosRedirectLogitProcessor(CustomLogitProcessor):
    """Guard ``<|im_end|>`` while the request is still inside ``<think>``.

    Per-request, two-level guard (applied on every decoding/verify step where
    ``<think>`` has been seen but ``</think>`` has not):

    1. **Ban**: ``<|im_end|>`` is forced to ``-inf`` on every row owned by the
       request, so no sampler (greedy / top-p / top-k / temperature) can ever
       emit it -- this closes the top-p/temperature leakage where
       ``<|im_end|>`` may sit in the top-p tail even when it isn't argmax.

    2. **Redirect**: on any row whose argmax is ``<|im_end|>`` (i.e. the model
       clearly wants to terminate), we additionally rewrite the row so the
       only viable token is ``</think>``. Once ``</think>`` is emitted the
       guard is released on subsequent steps automatically.

    Works on both code paths:
      * normal sampling  (``logits.shape[0] == len(custom_param_list)``)
      * spec v2 verify   (``logits.shape[0] == len(custom_param_list) * draft_token_num``)
    """

    THINKING_START_TOKEN_ID: int = 163606
    THINKING_END_TOKEN_ID: int = 163607

    IM_END_TOKEN_ID: int = 163586
    EOS_TOKEN_ID: int = 163585

    TOOL_CALL_SECTION_BEGIN_TOKEN_ID: int = 163595
    TOOL_CALL_SECTION_END_TOKEN_ID: int = 163596

    TOOL_CALL_BEGIN_TOKEN_ID: int = 163597
    TOOL_CALL_END_TOKEN_ID: int = 163599

    # Toggle the per-trigger debug print. Cheap: only fires when at least one
    # row actually gets redirected on this step.
    DEBUG: bool = False

    _ACTION_NONE: int = 0
    _ACTION_BAN_IM_END: int = 1
    _ACTION_REDIRECT_TO_THINK_END: int = 2

    def _is_inside_thinking(self, req: "Req") -> bool:
        """Return True iff <think> has been seen but </think> has not."""
        # </think> can appear either in the prompt (rare) or in the generated
        # tokens; check both to be safe.
        if (
            self.THINKING_END_TOKEN_ID in req.output_ids
            or self.THINKING_END_TOKEN_ID in req.origin_input_ids
        ):
            return False
        return (
            self.THINKING_START_TOKEN_ID in req.origin_input_ids
            or self.THINKING_START_TOKEN_ID in req.output_ids
        )

    def _has_valid_vocab(self, logits: torch.Tensor) -> bool:
        vocab_size = logits.shape[-1]
        required_token_ids = (
            self.IM_END_TOKEN_ID,
            self.THINKING_END_TOKEN_ID,
        )
        max_required_token_id = max(required_token_ids)
        if max_required_token_id < vocab_size:
            return True

        logger.warning(
            "[k25-eos-redirect] skip processor because vocab_size=%d does not cover "
            "required token ids %s",
            vocab_size,
            sorted(required_token_ids),
        )
        return False

    def _plan_actions_for_active_reqs(
        self,
        active_top_ids_cpu: torch.Tensor,
        tokens_per_req: int,
    ) -> torch.Tensor:
        actions = torch.zeros_like(active_top_ids_cpu, dtype=torch.int8)

        for req_row in range(active_top_ids_cpu.shape[0]):
            thinking_open = True
            for k in range(tokens_per_req):
                if not thinking_open:
                    break

                top = int(active_top_ids_cpu[req_row, k])
                if top == self.THINKING_END_TOKEN_ID:
                    thinking_open = False
                elif top == self.IM_END_TOKEN_ID:
                    actions[req_row, k] = self._ACTION_REDIRECT_TO_THINK_END
                    thinking_open = False
                else:
                    actions[req_row, k] = self._ACTION_BAN_IM_END

        return actions

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if not custom_param_list:
            return logits

        if not self._has_valid_vocab(logits):
            return logits

        n_rows = logits.shape[0]
        n_params = len(custom_param_list)
        # Standard sampling: tokens_per_req == 1.
        # Spec v2 verify:    tokens_per_req == draft_token_num.
        if n_params == 0 or n_rows % n_params != 0:
            return logits
        tokens_per_req = n_rows // n_params

        # ---- Stage 1: cheap CPU filter -- which requests are still inside
        # <think> at the start of this step?
        active_req_indices: List[int] = []
        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            req: Req = params.get("__req__")
            if req is None:
                continue
            if self._is_inside_thinking(req):
                active_req_indices.append(i)

        if not active_req_indices:
            return logits

        # ---- Stage 2: per-request sequential simulation across the
        # ``tokens_per_req`` draft rows.
        #
        # The previous version vectorised across all rows of a request and
        # treated every row as "still inside <think>". In MTP / spec v2 verify
        # this is wrong when the content is short: multiple draft positions of
        # the SAME request may simultaneously want to emit ``<|im_end|>`` (or
        # one position emits ``</think>`` and a later position emits
        # ``<|im_end|>``). Vectorised redirect then rewrites all of them,
        # producing accepted sequences such as ``12</think>12</think>12``.
        #
        # The correct semantics: within a single verify window the request
        # leaves the thinking stage as soon as ``</think>`` is emitted (either
        # naturally as the argmax, or after we redirect ``<|im_end|>`` to
        # ``</think>``). Subsequent rows in the same window must NOT be
        # guarded any more.
        #
        # draft_token_num is small (typically <= 8) so per-request iteration
        # is cheap. Keep the host/device round-trips narrow by only copying
        # argmax results for active requests, then upload one compact action
        # matrix back to the device.
        active_req_indices_t = torch.tensor(
            active_req_indices, device=logits.device, dtype=torch.long
        )
        active_top_ids_cpu = (
            logits.reshape(n_params, tokens_per_req, -1)
            .index_select(0, active_req_indices_t)
            .argmax(dim=-1)
            .cpu()
        )
        action_matrix = self._plan_actions_for_active_reqs(
            active_top_ids_cpu, tokens_per_req
        ).to(device=logits.device)
        row_offsets = active_req_indices_t[:, None] * tokens_per_req + torch.arange(
            tokens_per_req, device=logits.device
        )
        ban_rows = row_offsets[action_matrix == self._ACTION_BAN_IM_END]
        redirect_rows = row_offsets[
            action_matrix == self._ACTION_REDIRECT_TO_THINK_END
        ]

        # (a) Ban: suppress <|im_end|> on every still-in-think row.
        if ban_rows.numel() > 0:
            logits[ban_rows, self.IM_END_TOKEN_ID] = -float("inf")

        # (b) Redirect: full-mask + single-token release so ONLY </think> can
        # be emitted on that step, regardless of sampler configuration.
        if redirect_rows.numel() > 0:
            logits[redirect_rows, :] = -float("inf")
            logits[redirect_rows, self.THINKING_END_TOKEN_ID] = 0.0

            if self.DEBUG:
                print(
                    f"[k25-redirect] <|im_end|>-></think> "
                    f"redirect_rows={redirect_rows.tolist()} "
                    f"ban_rows={ban_rows.numel()} "
                    f"n_rows={n_rows} n_params={n_params} "
                    f"tokens_per_req={tokens_per_req}",
                    flush=True,
                )

        return logits



class KimiK25ReasoningEosRedirectTestLogitProcessor(CustomLogitProcessor):
    """Test variant: directional replacement of ``</think>`` with ``<|im_end|>``.

    For every row whose argmax token is ``</think>`` (i.e. the model is about to
    emit the thinking-end token at this step), this processor rewrites that
    single row so the only viable token is ``<|im_end|>``. All other rows pass
    through unchanged, so the rest of the request's behaviour is unaffected.

    Works on both code paths:
      * normal sampling (``logits.shape[0] == len(custom_param_list)``)
      * spec v2 verify (``logits.shape[0] == len(custom_param_list) * draft_token_num``)

    Purpose: end-to-end validation of the CLP plumbing (including
    ``eagle_info_v2.EagleVerifyInput.sample``). Not meant for production.
    """

    THINKING_END_TOKEN_ID: int = 163607
    EOS_TOKEN_ID: int = 163585
    IM_END_TOKEN_ID: int = 163586

    # Toggle the per-trigger debug print. Keep it cheap (only fires when at
    # least one row is being redirected on this step).
    DEBUG: bool = True

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if not custom_param_list:
            return logits

        # Vectorised "directional replace": find every row whose top-1 token is
        # </think> and rewrite just those rows to force <|im_end|>.
        top_ids = logits.argmax(dim=-1)
        redirect_mask = top_ids == self.THINKING_END_TOKEN_ID
        if not bool(redirect_mask.any()):
            return logits

        rows = redirect_mask.nonzero(as_tuple=True)[0]
        # Full-mask + single-token release: works for greedy AND for any
        # temperature/top-p/top-k sampler since only <|im_end|> is left.
        logits[rows, :] = -float("inf")
        logits[rows, self.IM_END_TOKEN_ID] = 0.0

        if self.DEBUG:
            n_rows = logits.shape[0]
            n_params = len(custom_param_list)
            # In spec verify n_rows == n_params * draft_token_num, so report
            # both to make path identification trivial in logs.
            print(
                f"[k25-test] redirect </think>-><|im_end|> rows={rows.tolist()} "
                f"n_rows={n_rows} n_params={n_params} "
                f"draft_token_num~={n_rows // max(n_params, 1)}",
                flush=True,
            )

        return logits


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence: List[int] = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = tuple(sequence[-(ngram_size - 1) :])
            else:
                current_prefix = tuple()

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits
