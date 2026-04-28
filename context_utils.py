from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RAG_PASSAGE_SEPARATOR = "\n\n"


def _tokenize_text(text: str, tokenizer, return_offsets_mapping: bool = False) -> Dict[str, Any]:
    if return_offsets_mapping:
        try:
            encoded = tokenizer([text], return_offsets_mapping=True)
        except TypeError:
            encoded = tokenizer(text, return_offsets_mapping=True)
    else:
        try:
            encoded = tokenizer([text])
        except TypeError:
            encoded = tokenizer(text)
    return encoded


def _flatten_token_ids(encoded: Dict[str, Any]) -> List[Any]:
    input_ids = encoded.get("input_ids", [])
    if len(input_ids) > 0 and isinstance(input_ids[0], list):
        return input_ids[0]
    return input_ids


def _flatten_offsets(encoded: Dict[str, Any]) -> List[Tuple[int, int]]:
    offsets = encoded.get("offset_mapping", [])
    if len(offsets) > 0 and isinstance(offsets[0], list):
        return offsets[0]
    return offsets


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(_flatten_token_ids(_tokenize_text(text, tokenizer)))


def drop_first_tokens(text: str, tokenizer, num_tokens: int) -> str:
    if num_tokens <= 0 or not text:
        return text

    encoded = _tokenize_text(text, tokenizer, return_offsets_mapping=True)
    token_ids = _flatten_token_ids(encoded)
    offsets = _flatten_offsets(encoded)
    if num_tokens >= len(token_ids):
        return ""
    if offsets:
        return text[offsets[num_tokens][0]:]

    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids[num_tokens:])
    raise ValueError("Tokenizer does not support offsets or decode for token truncation")


def keep_last_tokens(text: str, tokenizer, max_tokens: Optional[int]) -> str:
    if max_tokens is None:
        return text
    if max_tokens <= 0 or not text:
        return ""
    total_tokens = count_tokens(text, tokenizer)
    if total_tokens <= max_tokens:
        return text
    return drop_first_tokens(text, tokenizer, total_tokens - max_tokens)


def keep_first_tokens(text: str, tokenizer, max_tokens: Optional[int]) -> str:
    if max_tokens is None:
        return text
    if max_tokens <= 0 or not text:
        return ""
    encoded = _tokenize_text(text, tokenizer, return_offsets_mapping=True)
    token_ids = _flatten_token_ids(encoded)
    offsets = _flatten_offsets(encoded)
    if len(token_ids) <= max_tokens:
        return text
    if offsets:
        return text[:offsets[max_tokens][0]]
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids[:max_tokens])
    raise ValueError("Tokenizer does not support offsets or decode for token truncation")


def render_rag_passage(ctx: Dict[str, Any]) -> str:
    if ctx.get("title"):
        return "Document (Title: {title}): {text}".format(**ctx)
    return "Document: {text}".format(**ctx)


def render_rag_context(ctxs: Sequence[Dict[str, Any]]) -> str:
    return RAG_PASSAGE_SEPARATOR.join(render_rag_passage(ctx) for ctx in ctxs)


def build_contrast_test_item(
    test_item: Dict[str, Any],
    cd_mode: str,
    tokenizer=None,
    cd_shuffle_seed: int = 42,
    cd_window_tokens: Optional[int] = None,
    data: Optional[Dict[str, Any]] = None,
    max_length: Optional[int] = None,
    generation_max_length: Optional[int] = None,
    use_chat_template: bool = False,
    system_message: Optional[str] = None,
) -> Dict[str, Any]:
    contrast = dict(test_item)
    ctx = test_item.get("context", "") or ""
    if cd_mode == "cad":
        contrast["context"] = ""
    elif cd_mode == "shuffled":
        passages = ctx.split(RAG_PASSAGE_SEPARATOR)
        if len(passages) > 1:
            rng = random.Random(cd_shuffle_seed)
            rng.shuffle(passages)
        contrast["context"] = RAG_PASSAGE_SEPARATOR.join(passages)
    elif cd_mode == "reversed":
        passages = ctx.split(RAG_PASSAGE_SEPARATOR)
        contrast["context"] = RAG_PASSAGE_SEPARATOR.join(reversed(passages))
    elif cd_mode == "local_window":
        if tokenizer is None:
            raise ValueError("tokenizer is required for cd_mode=local_window")
        if cd_window_tokens is None or cd_window_tokens <= 0:
            raise ValueError("cd_window_tokens must be a positive integer for cd_mode=local_window")
        if data is not None and max_length is not None and generation_max_length is not None:
            ctx = truncate_context_for_prompt(
                contrast,
                data=data,
                tokenizer=tokenizer,
                max_length=max_length,
                generation_max_length=generation_max_length,
                use_chat_template=use_chat_template,
                system_message=system_message,
            )
        contrast["context"] = keep_last_tokens(ctx, tokenizer, cd_window_tokens)
    else:
        raise ValueError(f"Unknown cd_mode: {cd_mode!r}")
    return contrast


def apply_rag_context_mode(
    test_item: Dict[str, Any],
    context_mode: str = "off",
    tokenizer=None,
    context_window_tokens: Optional[int] = None,
    data: Optional[Dict[str, Any]] = None,
    max_length: Optional[int] = None,
    generation_max_length: Optional[int] = None,
    use_chat_template: bool = False,
    system_message: Optional[str] = None,
) -> Dict[str, Any]:
    transformed = dict(test_item)
    if context_mode == "off":
        return transformed
    if context_mode == "truncate_last":
        if tokenizer is None:
            raise ValueError("tokenizer is required for context_mode=truncate_last")
        if context_window_tokens is None or context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be a positive integer for truncate_last")
        context = transformed.get("context", "") or ""
        if data is not None and max_length is not None and generation_max_length is not None:
            context = truncate_context_for_prompt(
                transformed,
                data=data,
                tokenizer=tokenizer,
                max_length=max_length,
                generation_max_length=generation_max_length,
                use_chat_template=use_chat_template,
                system_message=system_message,
            )
        transformed["context"] = keep_last_tokens(context, tokenizer, context_window_tokens)
        return transformed
    if context_mode == "oracle_passages_only":
        oracle_ctxs = [dict(ctx) for ctx in transformed.get("ctxs", []) if ctx.get("has_answer")]
        transformed["ctxs"] = oracle_ctxs
        transformed["context"] = render_rag_context(oracle_ctxs)
        return transformed
    raise ValueError(f"Unknown context_mode: {context_mode!r}")


def _format_prompt(sample: Dict[str, Any], data: Dict[str, Any], use_chat_template: bool, system_message: Optional[str]) -> str:
    if use_chat_template:
        pieces = []
        if system_message:
            pieces.append(system_message)
        pieces.append(data["user_template"].format(**sample))
        return "\n".join(pieces)
    return data["prompt_template"].format(**sample)


def truncate_context_for_prompt(
    sample: Dict[str, Any],
    data: Dict[str, Any],
    tokenizer,
    max_length: int,
    generation_max_length: int,
    use_chat_template: bool = False,
    system_message: Optional[str] = None,
) -> str:
    budget = max_length - generation_max_length
    truncated = dict(sample)
    if budget <= 0:
        return ""

    while True:
        prompt = _format_prompt(truncated, data, use_chat_template, system_message)
        prompt_tokens = count_tokens(prompt, tokenizer)
        if prompt_tokens <= budget:
            return truncated.get("context", "") or ""
        overflow = prompt_tokens - budget
        current_context = truncated.get("context", "") or ""
        current_context_tokens = count_tokens(current_context, tokenizer)
        new_context = keep_first_tokens(current_context, tokenizer, current_context_tokens - overflow)
        if new_context == truncated.get("context", ""):
            return new_context
        truncated["context"] = new_context


def _passage_char_spans(ctxs: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for index, ctx in enumerate(ctxs):
        passage = render_rag_passage(ctx)
        start = cursor
        end = start + len(passage)
        spans.append((start, end))
        cursor = end
        if index != len(ctxs) - 1:
            cursor += len(RAG_PASSAGE_SEPARATOR)
    return spans


def _position_bucket(prefix_tokens: int, total_tokens: int) -> Optional[str]:
    if total_tokens <= 0:
        return None
    ratio = prefix_tokens / total_tokens
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "middle"
    return "late"


def compute_rag_example_metadata(
    sample: Dict[str, Any],
    data: Dict[str, Any],
    tokenizer,
    max_length: int,
    generation_max_length: int,
    use_chat_template: bool = False,
    system_message: Optional[str] = None,
    trailing_window_tokens: Iterable[int] = (2000, 8000),
) -> Dict[str, Any]:
    ctxs = sample.get("ctxs", []) or []
    full_context = render_rag_context(ctxs) if ctxs else sample.get("context", "") or ""
    sample_for_prompt = dict(sample)
    sample_for_prompt["context"] = full_context
    visible_context = truncate_context_for_prompt(
        sample_for_prompt,
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        generation_max_length=generation_max_length,
        use_chat_template=use_chat_template,
        system_message=system_message,
    )
    visible_tokens = count_tokens(visible_context, tokenizer)
    if not full_context:
        metadata = {
            "first_answer_passage_rank": None,
            "answer_position_bucket": None,
            "has_visible_answer": False,
            "visible_context_tokens": visible_tokens,
        }
        for window in trailing_window_tokens:
            metadata[f"gold_inside_window_{window}"] = False
        return metadata

    start_char = full_context.rfind(visible_context) if visible_context else len(full_context)
    if start_char < 0:
        start_char = max(0, len(full_context) - len(visible_context))

    answer_passages: List[Tuple[int, int, int, int, int]] = []
    spans = _passage_char_spans(ctxs)
    for idx, ctx in enumerate(ctxs):
        if not ctx.get("has_answer"):
            continue
        start, end = spans[idx]
        overlap_start = max(start, start_char)
        overlap_end = min(end, start_char + len(visible_context))
        if overlap_start < overlap_end:
            answer_passages.append((idx, start, end, overlap_start, overlap_end))

    metadata = {
        "first_answer_passage_rank": answer_passages[0][0] + 1 if answer_passages else None,
        "answer_position_bucket": None,
        "has_visible_answer": bool(answer_passages),
        "visible_context_tokens": visible_tokens,
    }

    if answer_passages and visible_context:
        _, passage_start, passage_end, overlap_start, overlap_end = answer_passages[0]
        midpoint_in_full = (max(passage_start, overlap_start) + min(passage_end, overlap_end)) // 2
        visible_answer_midpoint = max(0, midpoint_in_full - start_char)
        prefix_tokens = count_tokens(visible_context[:visible_answer_midpoint], tokenizer)
        metadata["answer_position_bucket"] = _position_bucket(prefix_tokens, visible_tokens)

    for window in trailing_window_tokens:
        window_suffix = keep_last_tokens(visible_context, tokenizer, window)
        suffix_start = len(visible_context) - len(window_suffix)
        suffix_start_in_full = start_char + max(0, suffix_start)
        metadata[f"gold_inside_window_{window}"] = any(
            overlap_end > suffix_start_in_full for _, _, _, _, overlap_end in answer_passages
        )

    return metadata
