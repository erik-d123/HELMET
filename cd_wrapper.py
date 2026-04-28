"""
Contrastive-decoding wrapper around HELMET's HFModel.

At each decoding step we compute logits twice:
    Pass A: the normal HELMET input (full context)
    Pass B: a contrast input that depends on --cd_mode:
        "cad"      -> empty context   (Shi et al., NAACL 2024)
        "shuffled" -> passages split on "\\n\\n" and shuffled (novel)
        "reversed" -> passages split on "\\n\\n" and reversed (shuffled-CD control)
        "local_window" -> keep only the last N context tokens before the question
and emit:  logits_final = logits_A - alpha * logits_B

Core contrast rule (one line) follows Shi et al., "Trusting Your Evidence:
Hallucinate Less with Context-Aware Decoding", https://arxiv.org/abs/2305.14739
Reference implementation (MIT): https://github.com/xhan77/context-aware-decoding
The rest of this file is HF transformers integration specific to HELMET.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import BatchEncoding, LogitsProcessor, LogitsProcessorList

from context_utils import build_contrast_test_item
from model_utils import tokenize

logger = logging.getLogger(__name__)


class ContrastiveLogitsProcessor(LogitsProcessor):
    """Per-step contrast against a separately maintained Pass-B KV cache.

    Held state between calls: the Pass-B KV cache, the last sampled token (to
    feed into Pass B on the next step), and (optionally) a per-step trace.
    """

    def __init__(
        self,
        base_causal_lm: torch.nn.Module,
        contrast_input_ids: torch.Tensor,
        contrast_attention_mask: Optional[torch.Tensor],
        alpha: float,
        log_trace: bool = False,
    ) -> None:
        self.model = base_causal_lm
        self.alpha = alpha
        self.log_trace = log_trace
        self.trace: List[Dict[str, Any]] = []

        device = next(base_causal_lm.parameters()).device
        contrast_input_ids = contrast_input_ids.to(device)
        if contrast_attention_mask is not None:
            contrast_attention_mask = contrast_attention_mask.to(device)

        # Prefill Pass B on [0..Lb-2] via the base transformer body so we do not
        # materialize logits over the whole prompt (the LM head over 128K tokens
        # would OOM on 80GB). Mirrors HFModel.generate's memory-saving prefill.
        base_body = getattr(base_causal_lm, "model", None)
        past_key_values_B = None
        if base_body is not None:
            with torch.no_grad():
                prefill_kwargs = dict(
                    input_ids=contrast_input_ids[:, :-1],
                    use_cache=True,
                    return_dict=True,
                )
                if contrast_attention_mask is not None:
                    prefill_kwargs["attention_mask"] = contrast_attention_mask[:, :-1]
                prefill_out = base_body(**prefill_kwargs)
                past_key_values_B = prefill_out.past_key_values

        if past_key_values_B is None:
            # Fallback: no prefill. On the first __call__ we will feed the full
            # Pass B prompt instead of just the last token.
            logger.warning("CD: Pass B prefill unavailable; feeding full prompt on step 0")
            self._next_b_token = contrast_input_ids
            self._b_attention_mask = contrast_attention_mask
        else:
            self._next_b_token = contrast_input_ids[:, -1:]
            self._b_attention_mask = None

        self.past_key_values_B = past_key_values_B
        self._first_call = True
        self._last_a_seq_len: Optional[int] = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        forward_kwargs: Dict[str, Any] = {"use_cache": True, "return_dict": True}
        if self._first_call:
            # Feed Pass B's final prompt token (or whole prompt if no prefill).
            forward_kwargs["input_ids"] = self._next_b_token
            if self._b_attention_mask is not None:
                forward_kwargs["attention_mask"] = self._b_attention_mask
            self._first_call = False
        else:
            new_tokens = input_ids[:, self._last_a_seq_len:]
            assert new_tokens.shape[1] == 1, (
                f"Expected exactly one new token per step, got {new_tokens.shape[1]}"
            )
            forward_kwargs["input_ids"] = new_tokens
        if self.past_key_values_B is not None:
            forward_kwargs["past_key_values"] = self.past_key_values_B
        self._last_a_seq_len = input_ids.shape[1]

        with torch.no_grad():
            b_out = self.model(**forward_kwargs)
        self.past_key_values_B = b_out.past_key_values
        scores_B = b_out.logits[:, -1, :]

        final = scores - self.alpha * scores_B

        if self.log_trace:
            self._record_step(scores, scores_B, final)
        return final

    def _record_step(self, scores_A: torch.Tensor, scores_B: torch.Tensor, final: torch.Tensor) -> None:
        top = 5
        def _top(t: torch.Tensor) -> List[List[float]]:
            v, i = torch.topk(t[0], top)
            return [[int(tok), float(val)] for tok, val in zip(i.tolist(), v.tolist())]
        self.trace.append({
            "step": len(self.trace),
            "logits_A_top": _top(scores_A),
            "logits_B_top": _top(scores_B),
            "logits_final_top": _top(final),
        })


class ContrastiveDecodingWrapper:
    """Wraps a HELMET HFModel. Non-overridden attributes delegate to the base."""

    _PASS_B_KEY_IDS = "_cd_pass_b_input_ids"
    _PASS_B_KEY_MASK = "_cd_pass_b_attention_mask"

    def __init__(
        self,
        base_model,
        cd_mode: str = "cad",
        cd_alpha: float = 1.0,
        cd_shuffle_seed: int = 42,
        cd_window_tokens: Optional[int] = None,
        cd_log_trace: bool = False,
        cd_trace_path: Optional[str] = None,
    ) -> None:
        if cd_mode not in {"cad", "shuffled", "reversed", "local_window"}:
            raise ValueError(f"Unknown cd_mode: {cd_mode!r}")
        self.base_model = base_model
        self.cd_mode = cd_mode
        self.cd_alpha = cd_alpha
        self.cd_shuffle_seed = cd_shuffle_seed
        self.cd_window_tokens = cd_window_tokens
        self.cd_log_trace = cd_log_trace
        self.cd_trace_path = cd_trace_path
        self._trace_fh = None
        if cd_log_trace and cd_trace_path is not None:
            self._trace_fh = open(cd_trace_path, "a")

    def __getattr__(self, name: str) -> Any:
        # Only called if the attribute is not found on self
        return getattr(self.__dict__["base_model"], name)

    # ---- inputs ----------------------------------------------------------

    def _build_contrast_test_item(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        return build_contrast_test_item(
            test_item,
            cd_mode=self.cd_mode,
            tokenizer=self.base_model.tokenizer,
            cd_shuffle_seed=self.cd_shuffle_seed,
            cd_window_tokens=self.cd_window_tokens,
            data=data,
            max_length=self.base_model.max_length,
            generation_max_length=self.base_model.generation_max_length,
            use_chat_template=self.base_model.use_chat_template,
            system_message=self.base_model.system_message,
        )

    def prepare_inputs(self, test_item: Dict[str, Any], data: Dict[str, Any]) -> BatchEncoding:
        # Pass A — exactly HELMET's normal tokenization
        inputs_a = self.base_model.prepare_inputs(test_item, data)

        # Pass B — re-tokenize with contrast context via the same helper
        contrast_item = self._build_contrast_test_item(test_item, data)
        inputs_b = tokenize(
            contrast_item,
            data,
            tokenizer=self.base_model.tokenizer,
            max_length=self.base_model.max_length,
            generation_max_length=self.base_model.generation_max_length,
            use_chat_template=self.base_model.use_chat_template,
            system_message=self.base_model.system_message,
        )

        merged = BatchEncoding({
            "input_ids": inputs_a["input_ids"],
            "attention_mask": inputs_a["attention_mask"],
            self._PASS_B_KEY_IDS: inputs_b["input_ids"],
            self._PASS_B_KEY_MASK: inputs_b["attention_mask"],
        })
        return merged

    # ---- generation ------------------------------------------------------

    @torch.no_grad()
    def generate(self, inputs: Optional[BatchEncoding] = None, prompt: Optional[str] = None, **kwargs):
        assert inputs is not None, "CD wrapper requires prepared inputs; prompt-only generation not supported"
        base = self.base_model
        device = base.model.device

        # Extract Pass B side inputs
        pass_b_input_ids = inputs.pop(self._PASS_B_KEY_IDS).to(device)
        pass_b_attention_mask = inputs.pop(self._PASS_B_KEY_MASK, None)
        if pass_b_attention_mask is not None:
            pass_b_attention_mask = pass_b_attention_mask.to(device)

        inputs = inputs.to(device)
        input_len = inputs["input_ids"].size(1)

        # Prefill Pass A (mirrors HFModel.generate)
        pass_a_inputs = inputs
        if hasattr(base.model, "model") and not base.disable_prefill:
            prefill = base.model.model(
                input_ids=inputs["input_ids"][..., :-1],
                attention_mask=inputs["attention_mask"][..., :-1],
            )
            past_kv_a = prefill.past_key_values
            if past_kv_a is None:
                base.disable_prefill = True
                logger.warning("past key values is None; disabling Pass A prefill")
            else:
                pass_a_inputs = BatchEncoding({
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "past_key_values": past_kv_a,
                })

        # Set up the contrastive processor (prefills Pass B internally)
        cd_processor = ContrastiveLogitsProcessor(
            base_causal_lm=base.model,
            contrast_input_ids=pass_b_input_ids,
            contrast_attention_mask=pass_b_attention_mask,
            alpha=self.cd_alpha,
            log_trace=self.cd_log_trace,
        )

        outputs = base.model.generate(
            **pass_a_inputs,
            max_new_tokens=base.generation_max_length,
            min_new_tokens=base.generation_min_length,
            do_sample=base.do_sample,
            temperature=base.temperature,
            top_p=base.top_p,
            eos_token_id=base.stop_token_ids,
            pad_token_id=base.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            logits_processor=LogitsProcessorList([cd_processor]),
        )

        text = base.tokenizer.decode(
            outputs["sequences"][0, input_len:], skip_special_tokens=True
        )
        save_prompt = (
            base.tokenizer.decode(pass_a_inputs["input_ids"][0][:500])
            + " <skip> "
            + base.tokenizer.decode(pass_a_inputs["input_ids"][0][-500:])
        )
        output_len = outputs["sequences"].size(1) - input_len

        if self.cd_log_trace and self._trace_fh is not None:
            for record in cd_processor.trace:
                self._trace_fh.write(json.dumps(record) + "\n")
            self._trace_fh.flush()

        del pass_a_inputs
        del outputs
        del cd_processor

        return {
            "output": text,
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
        }

    def generate_batch(self, inputs=None, prompt=None, **kwargs):
        from tqdm import tqdm
        assert inputs is not None, "CD wrapper expects pre-tokenized inputs"
        outputs = []
        for item in tqdm(inputs, desc=f"CD[{self.cd_mode},a={self.cd_alpha}]"):
            outputs.append(self.generate(inputs=item, **kwargs))
        return outputs

    def __del__(self):
        if getattr(self, "_trace_fh", None) is not None:
            try:
                self._trace_fh.close()
            except Exception:
                pass
