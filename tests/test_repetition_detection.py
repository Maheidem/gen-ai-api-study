"""
Tests for repetition detection utilities.
"""

import pytest
from local_llm_sdk.utils.repetition_detector import (
    detect_ngram_repetition,
    detect_token_repetition,
    calculate_entropy,
    is_anomalous,
    RepetitionDetector
)


class TestNgramRepetition:
    """Tests for n-gram repetition detection."""

    def test_detects_simple_repetition(self):
        """Should detect obvious repetition."""
        text = "the the the the the the the the"
        is_rep, ratio = detect_ngram_repetition(text, n=1, threshold=0.5)
        assert is_rep
        assert ratio > 0.8

    def test_detects_trigram_repetition(self):
        """Should detect repeated 3-word patterns."""
        text = "hello world test " * 5  # Pattern repeats 5 times
        is_rep, ratio = detect_ngram_repetition(text, n=3, threshold=0.5)
        assert is_rep

    def test_no_repetition_in_diverse_text(self):
        """Should not flag diverse text as repetitive."""
        text = "This is a completely unique sentence with no repeated patterns whatsoever"
        is_rep, ratio = detect_ngram_repetition(text, n=3, threshold=0.5)
        assert not is_rep
        assert ratio < 0.3

    def test_empty_text(self):
        """Should handle empty text gracefully."""
        is_rep, ratio = detect_ngram_repetition("", n=3, threshold=0.5)
        assert not is_rep
        assert ratio == 0.0

    def test_text_shorter_than_ngram(self):
        """Should handle text shorter than n-gram size."""
        is_rep, ratio = detect_ngram_repetition("hi", n=3, threshold=0.5)
        assert not is_rep
        assert ratio == 0.0


class TestTokenRepetition:
    """Tests for token sequence repetition detection."""

    def test_detects_repeated_sequences(self):
        """Should detect when token sequences repeat."""
        tokens = ["a", "b", "c"] * 5  # Pattern repeats 5 times
        is_rep = detect_token_repetition(tokens, window=3, max_repeats=3)
        assert is_rep

    def test_no_repetition_in_unique_sequence(self):
        """Should not flag unique sequences."""
        tokens = ["one", "two", "three", "four", "five", "six"]
        is_rep = detect_token_repetition(tokens, window=2, max_repeats=2)
        assert not is_rep

    def test_allows_some_repetition(self):
        """Should allow repetition below threshold."""
        tokens = ["a", "b"] * 2  # Only 2 repeats
        is_rep = detect_token_repetition(tokens, window=2, max_repeats=3)
        assert not is_rep  # Below max_repeats

    def test_short_token_list(self):
        """Should handle token lists shorter than window."""
        tokens = ["a", "b"]
        is_rep = detect_token_repetition(tokens, window=10, max_repeats=2)
        assert not is_rep


class TestEntropy:
    """Tests for entropy calculation."""

    def test_zero_entropy_for_repetition(self):
        """Repetitive text should have very low entropy."""
        text = "aaaaaaaaaa"
        entropy = calculate_entropy(text)
        assert entropy < 1.0  # Very low diversity

    def test_high_entropy_for_diverse_text(self):
        """Diverse text should have higher entropy."""
        text = "The quick brown fox jumps over the lazy dog"
        entropy = calculate_entropy(text)
        assert entropy > 3.0  # Natural language typically 3-5

    def test_empty_text_entropy(self):
        """Should handle empty text."""
        entropy = calculate_entropy("")
        assert entropy == 0.0


class TestAnomalyDetection:
    """Tests for statistical anomaly detection."""

    def test_detects_entropy_collapse(self):
        """Should detect when entropy drops significantly."""
        baseline = {
            "avg_length": 150,
            "avg_entropy": 4.0,
            "vocab_diversity": 0.75
        }

        # Very repetitive text
        text = "the the the the the the the the"
        is_anom, reason = is_anomalous(text, baseline)
        assert is_anom
        assert reason in ["ENTROPY_COLLAPSE", "LOW_DIVERSITY"]

    def test_detects_low_diversity(self):
        """Should detect when vocabulary diversity is low."""
        baseline = {
            "avg_length": 150,
            "avg_entropy": 4.0,
            "vocab_diversity": 0.75
        }

        # Same words repeated
        text = "test test test test test"
        is_anom, reason = is_anomalous(text, baseline)
        assert is_anom

    def test_detects_abnormal_length(self):
        """Should detect abnormally long responses."""
        baseline = {
            "avg_length": 100,
            "avg_entropy": 4.0,
            "vocab_diversity": 0.75
        }

        # 10x the baseline length with diverse words
        text = " ".join([f"word{i}" for i in range(1000)])
        is_anom, reason = is_anomalous(text, baseline)
        assert is_anom
        assert reason == "ABNORMAL_LENGTH"

    def test_no_anomaly_in_normal_text(self):
        """Should not flag normal text."""
        baseline = {
            "avg_length": 150,
            "avg_entropy": 4.0,
            "vocab_diversity": 0.75
        }

        text = "This is a perfectly normal response with good diversity and appropriate length for testing"
        is_anom, reason = is_anomalous(text, baseline)
        assert not is_anom


class TestRepetitionDetector:
    """Tests for the main RepetitionDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return RepetitionDetector(
            ngram_threshold=0.5,
            ngram_size=3,
            token_window=10,
            max_token_repeats=3
        )

    def test_detects_ngram_repetition(self, detector):
        """Should detect n-gram repetition."""
        text = "the the the the the the"
        is_rep, reason, metrics = detector.detect(text)
        assert is_rep
        assert reason == "NGRAM_REPETITION"
        assert "ngram_ratio" in metrics

    def test_detects_token_repetition(self, detector):
        """Should detect token sequence repetition."""
        text = "hello world " * 10
        is_rep, reason, metrics = detector.detect(text)
        assert is_rep
        assert reason in ["TOKEN_SEQUENCE_REPETITION", "NGRAM_REPETITION"]

    def test_detects_entropy_collapse(self, detector):
        """Should detect entropy-based anomalies."""
        text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        is_rep, reason, metrics = detector.detect(text)
        assert is_rep
        assert "ENTROPY" in reason or "DIVERSITY" in reason

    def test_no_detection_for_normal_text(self, detector):
        """Should not flag normal text."""
        text = "This is a completely normal response with good diversity"
        is_rep, reason, metrics = detector.detect(text)
        assert not is_rep
        assert reason == ""

    def test_updates_baseline_from_responses(self, detector):
        """Should update baseline metrics from successful responses."""
        successful = [
            "This is response one with good diversity",
            "Here is another different response",
            "A third unique response example"
        ]

        old_baseline = detector.baseline.copy()
        detector.update_baseline(successful)

        # Baseline should update
        assert detector.baseline != old_baseline
        assert detector.baseline["avg_length"] > 0
        assert detector.baseline["avg_entropy"] > 0

    def test_returns_metrics(self, detector):
        """Should return metrics dict."""
        text = "test response"
        is_rep, reason, metrics = detector.detect(text)
        assert isinstance(metrics, dict)
        assert "entropy" in metrics or "ngram_ratio" in metrics
