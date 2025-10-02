"""
Repetition detection utilities for LLM responses.

Detects various patterns of repetition that indicate model degradation:
- N-gram repetition (token-level loops)
- Token sequence repetition (block-level loops)
- Entropy collapse (loss of diversity)
- Statistical anomalies
"""

from collections import Counter
from typing import Tuple, Dict, List
import math


def detect_ngram_repetition(text: str, n: int = 3, threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Detect repetitive n-grams in generated text.

    Args:
        text: The text to analyze
        n: N-gram size (default: 3 for trigrams)
        threshold: Repetition ratio threshold (0.0-1.0)

    Returns:
        Tuple of (is_repetitive, repetition_ratio)

    Example:
        >>> detect_ngram_repetition("the the the the the")
        (True, 0.75)  # 75% repetitive
    """
    if not text or not text.strip():
        return False, 0.0

    words = text.split()

    if len(words) < n:
        return False, 0.0

    # Generate n-grams
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

    if not ngrams:
        return False, 0.0

    # Count occurrences
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    unique_ngrams = len(ngram_counts)

    # Calculate repetition ratio
    repetition_ratio = 1 - (unique_ngrams / total_ngrams)

    is_repetitive = repetition_ratio > threshold

    return is_repetitive, repetition_ratio


def detect_token_repetition(tokens: List[str], window: int = 10, max_repeats: int = 3) -> bool:
    """
    Detect if the same token sequence repeats consecutively.

    Args:
        tokens: List of tokens to check
        window: Size of sequence window to check
        max_repeats: Maximum allowed consecutive repeats

    Returns:
        True if repetition detected, False otherwise

    Example:
        >>> tokens = ["a", "b", "c"] * 5  # Pattern repeats 5 times
        >>> detect_token_repetition(tokens, window=3, max_repeats=3)
        True  # Exceeded max_repeats
    """
    if len(tokens) < window * 2:
        return False

    # Check last tokens for repetition
    last_window = tokens[-window:]
    previous_window = tokens[-2*window:-window]

    if last_window == previous_window:
        # Count how many times this pattern repeats
        repeat_count = 1
        offset = 2
        while (len(tokens) >= window * (offset + 1) and
               tokens[-window*offset:-window*(offset-1)] == last_window):
            repeat_count += 1
            offset += 1

        if repeat_count >= max_repeats:
            return True

    return False


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of text (measure of randomness/diversity).

    Lower entropy = more repetitive, less diverse
    Higher entropy = more random, more diverse

    Args:
        text: Text to analyze

    Returns:
        Entropy value (typically 0-5 for natural language)

    Example:
        >>> calculate_entropy("the the the the")
        0.0  # No diversity
        >>> calculate_entropy("diverse unique text here")
        4.2  # Good diversity
    """
    if not text or not text.strip():
        return 0.0

    # Count character frequencies
    char_counts = Counter(text.lower())
    total_chars = len(text)

    # Calculate Shannon entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)

    return entropy


def is_anomalous(response_text: str, baseline_metrics: Dict[str, float]) -> Tuple[bool, str]:
    """
    Detect statistical anomalies in response compared to baseline.

    Args:
        response_text: The response to check
        baseline_metrics: Dict with keys: avg_length, avg_entropy, vocab_diversity

    Returns:
        Tuple of (is_anomalous, reason)

    Example:
        >>> baseline = {"avg_length": 150, "avg_entropy": 4.0, "vocab_diversity": 0.75}
        >>> is_anomalous("the the the the the the", baseline)
        (True, "ENTROPY_COLLAPSE")
    """
    if not response_text or not baseline_metrics:
        return False, ""

    # Calculate response metrics
    response_length = len(response_text)
    response_entropy = calculate_entropy(response_text)

    words = response_text.split()
    if words:
        unique_words = len(set(words))
        total_words = len(words)
        vocab_diversity = unique_words / total_words
    else:
        vocab_diversity = 0.0

    # Check for anomalies (3 standard deviations = extreme outlier)
    # Using simplified thresholds for now

    # 1. Check entropy collapse
    if response_entropy < baseline_metrics.get("avg_entropy", 3.0) * 0.5:
        return True, "ENTROPY_COLLAPSE"

    # 2. Check vocabulary diversity
    if vocab_diversity < baseline_metrics.get("vocab_diversity", 0.7) * 0.5:
        return True, "LOW_DIVERSITY"

    # 3. Check abnormal length (10x baseline)
    avg_length = baseline_metrics.get("avg_length", 150)
    if response_length > avg_length * 10:
        return True, "ABNORMAL_LENGTH"

    # 4. Check for very short responses with low entropy
    if response_length < 20 and response_entropy < 2.0:
        return True, "DEGENERATE_SHORT"

    return False, ""


class RepetitionDetector:
    """
    Main detector class with configurable thresholds and baseline learning.
    """

    def __init__(self,
                 ngram_threshold: float = 0.5,
                 ngram_size: int = 3,
                 token_window: int = 10,
                 max_token_repeats: int = 3):
        """
        Initialize detector with thresholds.

        Args:
            ngram_threshold: Threshold for n-gram repetition (0.0-1.0)
            ngram_size: Size of n-grams to check
            token_window: Window size for token sequence detection
            max_token_repeats: Maximum allowed consecutive repeats
        """
        self.ngram_threshold = ngram_threshold
        self.ngram_size = ngram_size
        self.token_window = token_window
        self.max_token_repeats = max_token_repeats

        # Baseline metrics (learned from successful responses)
        self.baseline = {
            "avg_length": 150,
            "avg_entropy": 3.5,
            "vocab_diversity": 0.75
        }

    def detect(self, text: str) -> Tuple[bool, str, Dict]:
        """
        Run all detection methods on text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (is_repetitive, reason, metrics)
        """
        if not text or not text.strip():
            return False, "", {}

        metrics = {}

        # 1. N-gram repetition
        is_ngram_rep, ngram_ratio = detect_ngram_repetition(
            text,
            n=self.ngram_size,
            threshold=self.ngram_threshold
        )
        metrics["ngram_ratio"] = ngram_ratio

        if is_ngram_rep:
            return True, "NGRAM_REPETITION", metrics

        # 2. Token sequence repetition
        words = text.split()
        is_token_rep = detect_token_repetition(
            words,
            window=self.token_window,
            max_repeats=self.max_token_repeats
        )

        if is_token_rep:
            return True, "TOKEN_SEQUENCE_REPETITION", metrics

        # 3. Entropy and anomaly detection
        entropy = calculate_entropy(text)
        metrics["entropy"] = entropy

        is_anom, anom_reason = is_anomalous(text, self.baseline)

        if is_anom:
            return True, anom_reason, metrics

        return False, "", metrics

    def update_baseline(self, successful_responses: List[str]):
        """
        Update baseline metrics from successful responses.

        Args:
            successful_responses: List of responses that worked correctly
        """
        if not successful_responses:
            return

        lengths = [len(r) for r in successful_responses]
        entropies = [calculate_entropy(r) for r in successful_responses]

        diversities = []
        for r in successful_responses:
            words = r.split()
            if words:
                diversities.append(len(set(words)) / len(words))

        self.baseline["avg_length"] = sum(lengths) / len(lengths)
        self.baseline["avg_entropy"] = sum(entropies) / len(entropies)
        if diversities:
            self.baseline["vocab_diversity"] = sum(diversities) / len(diversities)
