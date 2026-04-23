import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


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

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if not custom_param_list:
            return logits

        n_rows = logits.shape[0]
        n_params = len(custom_param_list)
        # Standard sampling: tokens_per_req == 1.
        # Spec v2 verify:    tokens_per_req == draft_token_num.
        if n_params == 0 or n_rows % n_params != 0:
            return logits
        tokens_per_req = n_rows // n_params

        # ---- Stage 1: cheap CPU filter -- which requests are still inside
        # <think> and therefore eligible for redirection?
        active_flags: List[bool] = [False] * n_params
        any_active = False
        for i, params in enumerate(custom_param_list):
            if not params:
                continue
            req: Req = params.get("__req__")
            if req is None:
                continue
            if self._is_inside_thinking(req):
                active_flags[i] = True
                any_active = True

        if not any_active:
            return logits

        # ---- Stage 2: expand request-level active flags to row-level.
        # Each request owns ``tokens_per_req`` consecutive rows.
        device = logits.device
        active_per_req = torch.tensor(active_flags, device=device, dtype=torch.bool)
        active_rows = torch.repeat_interleave(active_per_req, tokens_per_req)

        # ---- Stage 3: decide which active rows should be "redirected"
        # (argmax == <|im_end|>, closed with </think>) vs only "banned"
        # (any other row on an active request, where we just suppress
        # <|im_end|> so it cannot leak through top-p / temperature).
        top_ids = logits.argmax(dim=-1)
        redirect_mask = (top_ids == self.IM_END_TOKEN_ID) & active_rows
        ban_only_mask = active_rows & ~redirect_mask

        # Fast path: nothing to do (no request inside <think> has a row that
        # needs touching -- effectively impossible if active_rows.any() since
        # banning always applies, but keep the guard for safety).
        if not bool(active_rows.any()):
            return logits

        # (a) Ban: suppress <|im_end|> on every active row. Cheap, vectorised.
        if bool(ban_only_mask.any()):
            ban_rows = ban_only_mask.nonzero(as_tuple=True)[0]
            logits[ban_rows, self.IM_END_TOKEN_ID] = -float("inf")

        # (b) Redirect: full-mask + single-token release so ONLY </think> can
        # be emitted on that step, regardless of sampler configuration.
        if bool(redirect_mask.any()):
            redirect_rows = redirect_mask.nonzero(as_tuple=True)[0]
            logits[redirect_rows, :] = -float("inf")
            logits[redirect_rows, self.THINKING_END_TOKEN_ID] = 0.0

            if self.DEBUG:
                print(
                    f"[k25-redirect] <|im_end|>-></think> "
                    f"redirect_rows={redirect_rows.tolist()} "
                    f"ban_rows={int(ban_only_mask.sum().item())} "
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
