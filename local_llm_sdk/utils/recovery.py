"""
Recovery strategies for invalid LLM responses.

Implements multiple recovery approaches when validation fails:
1. Correction prompts
2. History sanitization
3. Temperature override
4. Checkpoint rollback
5. Graceful abort
"""

import logging
from typing import List, Tuple, Optional, Any
from copy import deepcopy

from ..models import ChatMessage, ChatCompletion, create_chat_message

logger = logging.getLogger(__name__)


class RecoveryStrategy:
    """
    Base class for recovery strategies.
    """

    def __init__(self, client):
        """
        Initialize with client reference.

        Args:
            client: LocalLLMClient instance
        """
        self.client = client

    def attempt_recovery(self,
                        invalid_response: ChatCompletion,
                        messages: List[ChatMessage],
                        error_type: str) -> Tuple[bool, Any]:
        """
        Attempt to recover from invalid response.

        Args:
            invalid_response: The invalid response that failed validation
            messages: Current conversation history
            error_type: Type of validation error

        Returns:
            Tuple of (success, response_or_error)
        """
        raise NotImplementedError


class CorrectionPromptStrategy(RecoveryStrategy):
    """
    Strategy 1: Send correction prompt to guide model back to correct format.
    """

    CORRECTION_PROMPTS = {
        "XML_DRIFT": "Your last response used XML format. Please use JSON tool_calls array format as specified.",
        "INVALID_JSON": "Your last tool call had invalid JSON. Please provide valid JSON in the arguments field.",
        "NGRAM_REPETITION": "Your last response was repetitive. Please provide a concise, non-repetitive answer.",
        "UNKNOWN_FUNCTION": "You called an unknown function. Please only use the available tools.",
        "MISSING_FIELD": "Your last tool call was missing required fields. Please include all required parameters.",
        "EMPTY_RESPONSE": "Please provide a response. You cannot return empty content without tool calls.",
    }

    def attempt_recovery(self,
                        invalid_response: ChatCompletion,
                        messages: List[ChatMessage],
                        error_type: str) -> Tuple[bool, Any]:
        """
        Try to recover with correction prompt.

        Args:
            invalid_response: Invalid response
            messages: Conversation history
            error_type: Error type

        Returns:
            (success, new_response or error)
        """
        correction_msg = self.CORRECTION_PROMPTS.get(
            error_type,
            "Your last response had an error. Please try again with correct format."
        )

        logger.info(f"Attempting correction prompt for {error_type}")

        # Add correction prompt to history
        recovery_messages = messages.copy()
        recovery_messages.append(create_chat_message("user", correction_msg))

        try:
            # Retry with correction
            new_response = self.client.chat(
                messages=recovery_messages,
                use_tools=True,
                return_full_response=True,
                temperature=0.3  # Lower temp for more deterministic behavior
            )

            return True, new_response

        except Exception as e:
            logger.error(f"Correction prompt failed: {e}")
            return False, str(e)


class HistorySanitizationStrategy(RecoveryStrategy):
    """
    Strategy 2: Remove malformed message from history and retry.
    """

    def attempt_recovery(self,
                        invalid_response: ChatCompletion,
                        messages: List[ChatMessage],
                        error_type: str) -> Tuple[bool, Any]:
        """
        Remove invalid message and retry.

        Args:
            invalid_response: Invalid response
            messages: Conversation history
            error_type: Error type

        Returns:
            (success, new_response or error)
        """
        logger.info(f"Attempting history sanitization for {error_type}")

        # Don't include the invalid response in history
        # Just retry with the original messages
        try:
            new_response = self.client.chat(
                messages=messages,  # Original history without invalid response
                use_tools=True,
                return_full_response=True,
                temperature=0.2  # Even lower temp
            )

            return True, new_response

        except Exception as e:
            logger.error(f"History sanitization failed: {e}")
            return False, str(e)


class TemperatureOverrideStrategy(RecoveryStrategy):
    """
    Strategy 3: Retry with very low temperature for deterministic behavior.
    """

    def attempt_recovery(self,
                        invalid_response: ChatCompletion,
                        messages: List[ChatMessage],
                        error_type: str) -> Tuple[bool, Any]:
        """
        Retry with temperature=0.1 for maximum stability.

        Args:
            invalid_response: Invalid response
            messages: Conversation history
            error_type: Error type

        Returns:
            (success, new_response or error)
        """
        logger.info(f"Attempting temperature override for {error_type}")

        try:
            new_response = self.client.chat(
                messages=messages,
                use_tools=True,
                return_full_response=True,
                temperature=0.1,  # Near-deterministic
                max_tokens=1024  # Limit tokens to prevent runaway
            )

            return True, new_response

        except Exception as e:
            logger.error(f"Temperature override failed: {e}")
            return False, str(e)


class CheckpointRollbackStrategy(RecoveryStrategy):
    """
    Strategy 4: Rollback to last known-good conversation state.
    """

    def __init__(self, client, checkpoints: Optional[List[List[ChatMessage]]] = None):
        """
        Initialize with checkpoints.

        Args:
            client: LocalLLMClient
            checkpoints: List of conversation checkpoints
        """
        super().__init__(client)
        self.checkpoints = checkpoints or []

    def attempt_recovery(self,
                        invalid_response: ChatCompletion,
                        messages: List[ChatMessage],
                        error_type: str) -> Tuple[bool, Any]:
        """
        Rollback to last checkpoint and retry.

        Args:
            invalid_response: Invalid response
            messages: Current (corrupted) history
            error_type: Error type

        Returns:
            (success, new_response or error)
        """
        if not self.checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False, "No checkpoints"

        logger.info(f"Attempting checkpoint rollback for {error_type}")

        # Rollback to last checkpoint
        clean_messages = self.checkpoints[-1]

        try:
            new_response = self.client.chat(
                messages=clean_messages,
                use_tools=True,
                return_full_response=True,
                temperature=0.2
            )

            return True, new_response

        except Exception as e:
            logger.error(f"Checkpoint rollback failed: {e}")
            return False, str(e)


class GracefulAbortStrategy(RecoveryStrategy):
    """
    Strategy 5: Abort with clear error explanation (last resort).
    """

    def attempt_recovery(self,
                        invalid_response: ChatCompletion,
                        messages: List[ChatMessage],
                        error_type: str) -> Tuple[bool, Any]:
        """
        Abort gracefully with error details.

        Args:
            invalid_response: Invalid response
            messages: Conversation history
            error_type: Error type

        Returns:
            (False, error_message)
        """
        logger.error(f"All recovery attempts failed for {error_type}. Aborting gracefully.")

        error_msg = f"Unable to recover from {error_type}. The model generated an invalid response that could not be corrected."

        return False, error_msg


class RecoveryManager:
    """
    Manages recovery strategies and orchestrates recovery attempts.
    """

    def __init__(self, client, max_attempts: int = 3):
        """
        Initialize recovery manager.

        Args:
            client: LocalLLMClient instance
            max_attempts: Maximum recovery attempts (default: 3)
        """
        self.client = client
        self.max_attempts = max_attempts
        self.checkpoints: List[List[ChatMessage]] = []

        # Initialize strategies in order of preference
        self.strategies = [
            CorrectionPromptStrategy(client),
            HistorySanitizationStrategy(client),
            TemperatureOverrideStrategy(client),
            CheckpointRollbackStrategy(client, self.checkpoints),
            GracefulAbortStrategy(client)
        ]

    def save_checkpoint(self, messages: List[ChatMessage]):
        """
        Save conversation checkpoint.

        Args:
            messages: Current conversation to checkpoint
        """
        self.checkpoints.append(deepcopy(messages))

        # Keep only last 5 checkpoints to save memory
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)

        logger.debug(f"Checkpoint saved ({len(self.checkpoints)} total)")

    def recover(self,
                invalid_response: ChatCompletion,
                messages: List[ChatMessage],
                error_type: str) -> Tuple[bool, Any]:
        """
        Attempt recovery using available strategies.

        Args:
            invalid_response: The invalid response
            messages: Conversation history
            error_type: Type of validation error

        Returns:
            Tuple of (success, response_or_error)
        """
        logger.warning(f"Starting recovery for {error_type}")

        # Try each strategy
        for i, strategy in enumerate(self.strategies):
            if i >= self.max_attempts:
                logger.warning(f"Reached max recovery attempts ({self.max_attempts})")
                break

            strategy_name = strategy.__class__.__name__
            logger.info(f"Trying strategy {i+1}/{self.max_attempts}: {strategy_name}")

            success, result = strategy.attempt_recovery(invalid_response, messages, error_type)

            if success:
                logger.info(f"Recovery successful with {strategy_name}")
                return True, result

        # All strategies failed
        logger.error("All recovery strategies failed")
        return False, "All recovery attempts failed"
