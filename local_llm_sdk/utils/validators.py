"""
Response validation pipeline for LLM outputs.

Implements multi-stage validation:
1. FastValidator - Regex checks (10ms)
2. StructuralValidator - JSON/schema validation (50ms)
3. SemanticValidator - Entropy/similarity checks (100ms)
4. LLMJudge - Optional LLM-based validation (100-500ms)
"""

import json
import re
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

from ..models import ChatCompletion, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    error_type: str = ""
    details: str = ""

    @property
    def passed(self) -> bool:
        return self.is_valid


class FastValidator:
    """
    Fast regex-based validation (10ms).

    Catches obvious format drift like XML tags.
    """

    def __init__(self):
        # Known error patterns
        self.xml_patterns = [
            r'<tool_call>',
            r'</tool_call>',
            r'<function=',
            r'</function>',
            r'<parameter=',
        ]

    def validate(self, response: ChatCompletion) -> ValidationResult:
        """
        Fast validation checks.

        Args:
            response: ChatCompletion to validate

        Returns:
            ValidationResult
        """
        content = response.choices[0].message.content or ""

        # Check for XML-like tool call format (format drift)
        for pattern in self.xml_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    error_type="XML_DRIFT",
                    details=f"Detected XML format: {pattern}"
                )

        # Check for basic JSON validity in tool_calls
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    # Try to parse arguments as JSON
                    json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError) as e:
                    return ValidationResult(
                        is_valid=False,
                        error_type="INVALID_JSON",
                        details=f"Tool call arguments not valid JSON: {e}"
                    )

        return ValidationResult(is_valid=True)


class StructuralValidator:
    """
    Structural validation against schemas (50ms).

    Validates JSON structure and required fields.
    """

    def __init__(self, tool_schemas: Optional[List[Dict]] = None):
        """
        Initialize with tool schemas.

        Args:
            tool_schemas: List of tool schemas to validate against
        """
        self.tool_schemas = {
            schema["function"]["name"]: schema["function"]
            for schema in (tool_schemas or [])
        }

    def validate(self, response: ChatCompletion) -> ValidationResult:
        """
        Validate response structure.

        Args:
            response: ChatCompletion to validate

        Returns:
            ValidationResult
        """
        message = response.choices[0].message

        # Check if tool_calls exist
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            # No tool calls is OK
            return ValidationResult(is_valid=True)

        # Validate each tool call
        for tc in message.tool_calls:
            function_name = tc.function.name

            # Check if function is known
            if self.tool_schemas and function_name not in self.tool_schemas:
                return ValidationResult(
                    is_valid=False,
                    error_type="UNKNOWN_FUNCTION",
                    details=f"Unknown function: {function_name}"
                )

            # Parse arguments
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    is_valid=False,
                    error_type="INVALID_JSON",
                    details=f"Cannot parse arguments: {e}"
                )

            # Check required fields if schema available
            if function_name in self.tool_schemas:
                schema = self.tool_schemas[function_name]
                params = schema.get("parameters", {})
                required = params.get("required", [])

                for field in required:
                    if field not in args:
                        return ValidationResult(
                            is_valid=False,
                            error_type="MISSING_FIELD",
                            details=f"Missing required field '{field}' in {function_name}"
                        )

        return ValidationResult(is_valid=True)


class SemanticValidator:
    """
    Semantic validation using repetition detection (100ms).

    Uses RepetitionDetector to check for semantic issues.
    """

    def __init__(self, repetition_detector=None):
        """
        Initialize with repetition detector.

        Args:
            repetition_detector: RepetitionDetector instance
        """
        from .repetition_detector import RepetitionDetector
        self.detector = repetition_detector or RepetitionDetector()

    def validate(self, response: ChatCompletion) -> ValidationResult:
        """
        Validate semantic quality of response.

        Args:
            response: ChatCompletion to validate

        Returns:
            ValidationResult
        """
        content = response.choices[0].message.content or ""

        if not content.strip():
            # Empty content is OK if there are tool calls
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return ValidationResult(is_valid=True)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_type="EMPTY_RESPONSE",
                    details="No content and no tool calls"
                )

        # Check for repetition
        is_rep, reason, metrics = self.detector.detect(content)

        if is_rep:
            return ValidationResult(
                is_valid=False,
                error_type=reason,
                details=f"Repetition detected: {metrics}"
            )

        return ValidationResult(is_valid=True)


class LLMJudge:
    """
    LLM-based validation using lightweight model (100-500ms).

    Uses a separate lightweight model to validate primary model's output.
    Optional - only use when enabled and suspicious patterns detected.
    """

    def __init__(self, judge_client=None, enabled: bool = False):
        """
        Initialize LLM judge.

        Args:
            judge_client: LocalLLMClient instance for judging (uses lightweight model)
            enabled: Whether judge is enabled
        """
        self.judge_client = judge_client
        self.enabled = enabled

    def validate(self, response: ChatCompletion, expected_schema: Optional[str] = None) -> ValidationResult:
        """
        Use LLM to validate response.

        Args:
            response: ChatCompletion to validate
            expected_schema: Optional schema description

        Returns:
            ValidationResult
        """
        if not self.enabled or not self.judge_client:
            # Judge disabled - pass through
            return ValidationResult(is_valid=True)

        # Build judge prompt
        response_json = response.model_dump_json(indent=2)

        judge_prompt = f"""You are a validator. Analyze this LLM response.

Expected format: OpenAI function calling (JSON with tool_calls array)
{f"Expected schema: {expected_schema}" if expected_schema else ""}

Response to validate:
{response_json[:500]}  # Truncate for speed

Answer ONLY with one word:
- VALID: Perfectly follows expected format
- INVALID: Has format errors or drift
- AMBIGUOUS: Unclear

Your verdict:"""

        try:
            # Get judgment (use low temp for deterministic verdict)
            verdict = self.judge_client.chat(
                judge_prompt,
                temperature=0.0,
                max_tokens=10
            )

            verdict_clean = str(verdict).strip().upper()

            if "VALID" in verdict_clean:
                return ValidationResult(is_valid=True)
            elif "INVALID" in verdict_clean:
                return ValidationResult(
                    is_valid=False,
                    error_type="JUDGE_REJECTED",
                    details="LLM judge detected format error"
                )
            else:
                # AMBIGUOUS - treat as invalid to be safe
                return ValidationResult(
                    is_valid=False,
                    error_type="JUDGE_UNCERTAIN",
                    details="LLM judge uncertain about validity"
                )

        except Exception as e:
            logger.error(f"LLM judge error: {e}")
            # Judge failed - pass through (don't block on judge errors)
            return ValidationResult(is_valid=True)


class ValidationPipeline:
    """
    Multi-stage validation pipeline.

    Runs validators in sequence, fail-fast on first error.
    """

    def __init__(self,
                 tool_schemas: Optional[List[Dict]] = None,
                 repetition_detector=None,
                 judge_client=None,
                 enable_judge: bool = False):
        """
        Initialize validation pipeline.

        Args:
            tool_schemas: Tool schemas for structural validation
            repetition_detector: RepetitionDetector instance
            judge_client: Client for LLM judge
            enable_judge: Whether to enable LLM judge
        """
        self.fast = FastValidator()
        self.structural = StructuralValidator(tool_schemas)
        self.semantic = SemanticValidator(repetition_detector)
        self.judge = LLMJudge(judge_client, enable_judge)

        self.enable_judge = enable_judge

    def should_use_judge(self, response: ChatCompletion, model_name: str = "") -> bool:
        """
        Decide if LLM judge should be used.

        Args:
            response: Response to check
            model_name: Name of model that generated response

        Returns:
            True if judge should be used
        """
        if not self.enable_judge:
            return False

        # Use judge for known problematic models
        if "qwen" in model_name.lower():
            return True

        # Could add more heuristics here
        # - Low entropy in response
        # - Unusual token count
        # - Random sampling for monitoring

        return False

    def validate_all(self, response: ChatCompletion, model_name: str = "") -> Tuple[bool, str]:
        """
        Run all validation stages.

        Args:
            response: ChatCompletion to validate
            model_name: Name of model (for judge triggering)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Stage 1: Fast validation
        result = self.fast.validate(response)
        if not result.passed:
            logger.warning(f"Fast validation failed: {result.error_type} - {result.details}")
            return False, result.error_type

        # Stage 2: Structural validation
        result = self.structural.validate(response)
        if not result.passed:
            logger.warning(f"Structural validation failed: {result.error_type} - {result.details}")
            return False, result.error_type

        # Stage 3: Semantic validation
        result = self.semantic.validate(response)
        if not result.passed:
            logger.warning(f"Semantic validation failed: {result.error_type} - {result.details}")
            return False, result.error_type

        # Stage 4: LLM judge (optional, only if suspicious)
        if self.should_use_judge(response, model_name):
            result = self.judge.validate(response)
            if not result.passed:
                logger.warning(f"LLM judge failed: {result.error_type} - {result.details}")
                return False, result.error_type

        return True, ""
