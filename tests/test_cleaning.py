"""Unit tests for aft.cleaning — dataset cleaning utilities."""

from __future__ import annotations

from unittest.mock import patch

from aft.cleaning import clean_dataset
from tests.conftest import FakeDataset, FakeTokenizer, make_fake_dataset


class TestWhitespaceNormalization:
    def test_trailing_spaces_removed(self) -> None:
        ds = make_fake_dataset(["hello world   ", "test\n\n\n\nend"])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, min_tokens=1)
        texts = result["text"]
        assert texts[0] == "hello world"
        assert "\n\n\n" not in texts[1]

    def test_triple_newlines_collapsed(self) -> None:
        ds = make_fake_dataset(["a\n\n\n\n\n\nb"])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, min_tokens=1)
        assert result["text"][0] == "a\n\nb"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        ds = make_fake_dataset(["  hello  "])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, min_tokens=1)
        assert result["text"][0] == "hello"

    def test_trailing_tabs_removed(self) -> None:
        ds = make_fake_dataset(["hello\t\t"])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, min_tokens=1)
        assert result["text"][0] == "hello"


class TestSpecialCharFilter:
    def test_normal_text_passes(self) -> None:
        ds = make_fake_dataset(["This is a normal sentence with words."])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, max_special_ratio=0.3, min_tokens=1)
        assert len(result) == 1

    def test_high_special_ratio_dropped(self) -> None:
        ds = make_fake_dataset(["!@#$%^&*()_+!!!"])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, max_special_ratio=0.3, min_tokens=1)
        assert len(result) == 0

    def test_empty_text_dropped(self) -> None:
        ds = make_fake_dataset([""])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, min_tokens=1)
        assert len(result) == 0

    def test_mixed_text_filtered_correctly(self) -> None:
        # 50% special chars
        ds = make_fake_dataset(["ab!@#$%"])
        tokenizer = FakeTokenizer()
        # Special ratio = 5/7 ≈ 0.71 > 0.3 → dropped
        result = clean_dataset(ds, tokenizer, max_special_ratio=0.3, min_tokens=1)
        assert len(result) == 0


class TestTokenLengthFilter:
    def test_within_bounds_kept(self) -> None:
        ds = make_fake_dataset(["hello world"])
        tokenizer = FakeTokenizer(tokens_per_char=1)
        result = clean_dataset(ds, tokenizer, min_tokens=1, max_tokens=100)
        assert len(result) == 1

    def test_too_short_dropped(self) -> None:
        ds = make_fake_dataset(["hi"])
        tokenizer = FakeTokenizer(tokens_per_char=1)
        # "hi" → 2 tokens, min_tokens=10 → dropped
        result = clean_dataset(ds, tokenizer, min_tokens=10, max_tokens=100)
        assert len(result) == 0

    def test_too_long_dropped(self) -> None:
        ds = make_fake_dataset(["a" * 200])
        tokenizer = FakeTokenizer(tokens_per_char=1)
        # 200 tokens, max_tokens=50 → dropped
        result = clean_dataset(ds, tokenizer, min_tokens=1, max_tokens=50)
        assert len(result) == 0

    def test_boundary_exact_match(self) -> None:
        ds = make_fake_dataset(["a" * 10])
        tokenizer = FakeTokenizer(tokens_per_char=1)
        result = clean_dataset(ds, tokenizer, min_tokens=10, max_tokens=10)
        assert len(result) == 1


class TestDeduplication:
    def test_duplicates_removed(self) -> None:
        texts = [
            "hello world test extra",
            "hello world test extra",
            "another text here ok",
        ]
        ds = make_fake_dataset(texts)
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, dedup=True, min_tokens=1)
        assert len(result) == 2
        assert result["text"][0] == "hello world test extra"
        assert result["text"][1] == "another text here ok"

    def test_all_unique_kept(self) -> None:
        ds = make_fake_dataset(["alpha text one", "beta text two", "gamma text three"])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, dedup=True, min_tokens=1)
        assert len(result) == 3

    def test_dedup_deterministic(self) -> None:
        """After the hashlib fix, dedup should be deterministic across runs."""
        texts = ["same text here ok", "same text here ok", "different text here"]
        ds1 = make_fake_dataset(texts)
        ds2 = make_fake_dataset(texts)
        tokenizer = FakeTokenizer()
        r1 = clean_dataset(ds1, tokenizer, dedup=True, min_tokens=1)
        r2 = clean_dataset(ds2, tokenizer, dedup=True, min_tokens=1)
        assert r1["text"] == r2["text"]

    def test_dedup_disabled(self) -> None:
        ds = make_fake_dataset(["hello world test ok", "hello world test ok"])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, dedup=False, min_tokens=1)
        assert len(result) == 2


class TestLanguageFilter:
    def test_language_filter_skipped_when_not_installed(self) -> None:
        """When langdetect is not installed, filter is skipped with warning."""
        import builtins

        ds = make_fake_dataset(["hello world test data"])
        tokenizer = FakeTokenizer()
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "langdetect":
                raise ImportError("No module named 'langdetect'")
            return real_import(name, *args, **kwargs)  # type: ignore[no-any-return]

        with patch.object(builtins, "__import__", side_effect=mock_import):
            result = clean_dataset(ds, tokenizer, languages=["en"], min_tokens=1)
        # Should not crash; filter skipped, all samples kept
        assert len(result) == 1


class TestCleanDatasetEdgeCases:
    def test_no_changes_needed(self) -> None:
        ds = make_fake_dataset(["Hello world, this is fine and long enough text."])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer, min_tokens=1)
        assert len(result) == 1

    def test_empty_dataset(self) -> None:
        ds = FakeDataset([])
        tokenizer = FakeTokenizer()
        result = clean_dataset(ds, tokenizer)
        assert len(result) == 0

    def test_multiple_cleaning_steps_combined(self) -> None:
        ds = make_fake_dataset(
            [
                "Hello world this is a test sentence.",  # clean
                "!!!@@@",  # special chars → dropped
                "hi",  # too short → dropped
                "Hello world this is a test sentence.",  # duplicate
            ]
        )
        tokenizer = FakeTokenizer(tokens_per_char=1)
        result = clean_dataset(
            ds, tokenizer, dedup=True, min_tokens=10, max_special_ratio=0.3
        )
        assert len(result) == 1
        assert result["text"][0] == "Hello world this is a test sentence."
