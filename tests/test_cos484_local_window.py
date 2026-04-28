import importlib
import types
import sys
import tempfile
import unittest
from pathlib import Path

from analysis_utils import bootstrap_macro_average, bootstrap_macro_delta
from context_utils import (
    apply_rag_context_mode,
    build_contrast_test_item,
    compute_rag_example_metadata,
    keep_last_tokens,
)


class WhitespaceTokenizer:
    is_fast = True

    def __call__(self, text_or_texts, return_offsets_mapping=False, **kwargs):
        if isinstance(text_or_texts, list):
            return self._encode(text_or_texts[0], nested=True, return_offsets_mapping=return_offsets_mapping)
        return self._encode(text_or_texts, nested=False, return_offsets_mapping=return_offsets_mapping)

    def _encode(self, text, nested, return_offsets_mapping=False):
        input_ids = []
        offsets = []
        idx = 0
        for token in text.split():
            start = text.index(token, idx)
            end = start + len(token)
            idx = end
            input_ids.append(token)
            offsets.append((start, end))
        if nested:
            output = {"input_ids": [input_ids]}
            if return_offsets_mapping:
                output["offset_mapping"] = [offsets]
            return output
        output = {"input_ids": input_ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output


class Cos484LocalWindowTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WhitespaceTokenizer()
        self.qa_data = {
            "prompt_template": "{context}\n\nQuestion: {question}\nAnswer:",
            "user_template": "{context}\n\nQuestion: {question}",
            "system_template": "Answer:",
        }
        self.sample = {
            "question": "Where is the answer?",
            "answer": ["middle"],
            "ctxs": [
                {"title": "Doc1", "text": "alpha beta gamma delta", "has_answer": False},
                {"title": "Doc2", "text": "middle answer evidence here", "has_answer": True},
                {"title": "Doc3", "text": "tail tokens stay local now", "has_answer": False},
            ],
            "context": (
                "Document (Title: Doc1): alpha beta gamma delta\n\n"
                "Document (Title: Doc2): middle answer evidence here\n\n"
                "Document (Title: Doc3): tail tokens stay local now"
            ),
        }

    def test_keep_last_tokens_returns_suffix(self):
        self.assertEqual(keep_last_tokens("zero one two three", self.tokenizer, 2), "two three")
        self.assertEqual(keep_last_tokens("zero one two three", self.tokenizer, 10), "zero one two three")
        self.assertEqual(keep_last_tokens("zero one two three", self.tokenizer, 0), "")

    def test_local_window_mode_crops_only_context_suffix(self):
        contrast = build_contrast_test_item(
            {"context": "zero one two three four"},
            cd_mode="local_window",
            tokenizer=self.tokenizer,
            cd_window_tokens=3,
        )
        self.assertEqual(contrast["context"], "two three four")

    def test_oracle_passages_only_keeps_answer_documents(self):
        transformed = apply_rag_context_mode(
            self.sample,
            context_mode="oracle_passages_only",
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(transformed["ctxs"]), 1)
        self.assertTrue(transformed["ctxs"][0]["has_answer"])
        self.assertIn("Doc2", transformed["context"])
        self.assertNotIn("Doc1", transformed["context"])

    def test_truncate_last_uses_same_suffix_logic(self):
        transformed = apply_rag_context_mode(
            self.sample,
            context_mode="truncate_last",
            tokenizer=self.tokenizer,
            context_window_tokens=4,
        )
        self.assertEqual(transformed["context"], "tokens stay local now")

    def test_rag_metadata_tracks_position_and_windows(self):
        metadata = compute_rag_example_metadata(
            self.sample,
            data=self.qa_data,
            tokenizer=self.tokenizer,
            max_length=256,
            generation_max_length=16,
            use_chat_template=False,
            system_message=None,
            trailing_window_tokens=(4, 12),
        )
        self.assertEqual(metadata["first_answer_passage_rank"], 2)
        self.assertEqual(metadata["answer_position_bucket"], "middle")
        self.assertFalse(metadata["gold_inside_window_4"])
        self.assertTrue(metadata["gold_inside_window_12"])

    def test_bootstrap_macro_average_is_deterministic_for_constant_scores(self):
        summary = bootstrap_macro_average(
            {
                "nq": [1.0, 1.0],
                "triviaqa": [0.0, 0.0],
                "hotpotqa": [1.0, 1.0],
                "popqa": [0.0, 0.0],
            },
            num_bootstrap_samples=200,
            seed=7,
        )
        self.assertEqual(summary["point_estimate"], 0.5)
        self.assertEqual(summary["ci_low"], 0.5)
        self.assertEqual(summary["ci_high"], 0.5)

    def test_bootstrap_macro_delta_is_zero_for_identical_scores(self):
        summary = bootstrap_macro_delta(
            {
                "nq": [1.0, 0.0],
                "triviaqa": [0.0, 1.0],
            },
            {
                "nq": [1.0, 0.0],
                "triviaqa": [0.0, 1.0],
            },
            num_bootstrap_samples=200,
            seed=11,
        )
        self.assertEqual(summary["point_estimate"], 0.0)
        self.assertEqual(summary["ci_low"], 0.0)
        self.assertEqual(summary["ci_high"], 0.0)

    def test_parse_arguments_accepts_attention_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("model_name_or_path: meta-llama/Llama-3.1-8B-Instruct\n")
            argv = sys.argv
            saved_yaml = sys.modules.get("yaml")
            try:
                fake_yaml = types.ModuleType("yaml")
                fake_yaml.safe_load = lambda fh: {
                    "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
                }
                sys.modules["yaml"] = fake_yaml
                arguments = importlib.import_module("arguments")
                sys.argv = [
                    "prog",
                    "--config",
                    str(config_path),
                    "--attn_implementation",
                    "sdpa",
                ]
                args = arguments.parse_arguments()
            finally:
                sys.argv = argv
                if saved_yaml is None:
                    sys.modules.pop("yaml", None)
                else:
                    sys.modules["yaml"] = saved_yaml
        self.assertEqual(args.attn_implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
