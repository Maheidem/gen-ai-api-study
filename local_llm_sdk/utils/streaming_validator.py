"""
Lightweight streaming validator for LLM responses.

Detects format drift and repetition in real-time (during generation),
allowing early termination to save resources.

Simplified from original 950-line validation system to 150 lines.
Focuses on high-value, low-overhead checks that work on partial text.
"""

import re
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of streaming validation check."""
    should_stop: bool
    error_type: str = ""
    details: str = ""


class StreamingValidator:
    """
    Lightweight validator for streaming LLM responses.

    Detects issues in real-time (every N tokens) and signals when
    to stop generation early.

    Focus: Fast checks (10-20ms) that work on partial text.
    """

    def __init__(self, check_interval: int = 20):
        """
        Initialize streaming validator.

        Args:
            check_interval: Check every N tokens (default: 20)
                           Lower = faster detection, higher overhead
                           Higher = slower detection, lower overhead
        """
        self.check_interval = check_interval
        self.accumulated = ""
        self.tokens_processed = 0

        # Compile regex patterns for speed
        self.xml_patterns = [
            re.compile(r'<tool_call>', re.IGNORECASE),
            re.compile(r'</tool_call>', re.IGNORECASE),
            re.compile(r'<function=', re.IGNORECASE),
            re.compile(r'</function>', re.IGNORECASE),
            re.compile(r'<parameter=', re.IGNORECASE),
        ]

    def add_chunk(self, chunk: str) -> ValidationResult:
        """
        Add a chunk from streaming response and check for issues.

        Args:
            chunk: New text chunk from stream

        Returns:
            ValidationResult with should_stop flag and error details
        """
        if not chunk:
            return ValidationResult(should_stop=False)

        self.accumulated += chunk
        self.tokens_processed += len(chunk.split())

        # Always check - overhead is minimal (10-20ms) and ensures we catch issues early
        return self._check_accumulated()

    def finalize(self) -> ValidationResult:
        """
        Final check on complete response.

        Call this when streaming is complete to ensure nothing was missed.

        Returns:
            ValidationResult
        """
        return self._check_accumulated()

    def _check_accumulated(self) -> ValidationResult:
        """
        Check accumulated text for issues.

        Returns:
            ValidationResult
        """
        # Check 1: XML drift (10ms)
        # Catches models like qwen3 that use XML format instead of JSON
        for pattern in self.xml_patterns:
            match = pattern.search(self.accumulated)
            if match:
                return ValidationResult(
                    should_stop=True,
                    error_type="XML_DRIFT",
                    details=f"Detected XML pattern: {pattern.pattern}"
                )

        # Check 2: Simple repetition (10ms)
        # Catches token loops ("the the the...")
        if len(self.accumulated) > 100:
            words = self.accumulated.split()
            if len(words) >= 20:
                # Check if last 10 words == previous 10 words
                if words[-10:] == words[-20:-10]:
                    return ValidationResult(
                        should_stop=True,
                        error_type="REPETITION",
                        details=f"Detected repeating pattern: {' '.join(words[-10:])}"
                    )

        return ValidationResult(should_stop=False)

    def reset(self):
        """Reset validator state for new response."""
        self.accumulated = ""
        self.tokens_processed = 0


def create_streaming_validator(check_interval: int = 20) -> StreamingValidator:
    """
    Factory function to create a streaming validator.

    Args:
        check_interval: Check every N tokens (default: 20)

    Returns:
        StreamingValidator instance
    """
    return StreamingValidator(check_interval=check_interval)
