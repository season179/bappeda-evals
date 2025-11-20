"""
Error handling and retry logic for API calls

Provides robust error classification and retry strategies.
"""

import functools
import time
from typing import Any, Callable, List, Optional, Tuple, Type

from openai import (
    APIError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError
)

from .logger import get_logger


logger = get_logger(__name__)


class ErrorClassifier:
    """Classifies API errors and determines retry strategy"""

    RETRYABLE_ERRORS = [
        RateLimitError,  # 429
        APIError,        # 500-503
        APIConnectionError,  # Network errors
        APITimeoutError  # Timeout errors
    ]

    FATAL_ERRORS = [
        AuthenticationError,  # 401
        PermissionDeniedError,  # 403
        NotFoundError  # 404
    ]

    @staticmethod
    def should_retry(error: Exception) -> bool:
        """
        Determine if an error should be retried

        Args:
            error: Exception to classify

        Returns:
            True if error is retryable
        """
        error_type = type(error)
        return any(isinstance(error, err_type) for err_type in ErrorClassifier.RETRYABLE_ERRORS)

    @staticmethod
    def is_fatal(error: Exception) -> bool:
        """
        Determine if an error is fatal (non-retryable)

        Args:
            error: Exception to classify

        Returns:
            True if error is fatal
        """
        error_type = type(error)
        return any(isinstance(error, err_type) for err_type in ErrorClassifier.FATAL_ERRORS)

    @staticmethod
    def get_error_category(error: Exception) -> str:
        """
        Get human-readable error category

        Args:
            error: Exception to classify

        Returns:
            Error category name
        """
        if isinstance(error, AuthenticationError):
            return "Authentication Error (401)"
        elif isinstance(error, PermissionDeniedError):
            return "Permission Denied (403)"
        elif isinstance(error, NotFoundError):
            return "Not Found (404)"
        elif isinstance(error, RateLimitError):
            return "Rate Limit (429)"
        elif isinstance(error, APIConnectionError):
            return "Connection Error"
        elif isinstance(error, APITimeoutError):
            return "Timeout Error"
        elif isinstance(error, APIError):
            return "API Error (500+)"
        else:
            return "Unknown Error"


def classify_error(error: Exception) -> Tuple[str, bool, bool]:
    """
    Classify an error and determine retry strategy

    Args:
        error: Exception to classify

    Returns:
        Tuple of (category, should_retry, is_fatal)
    """
    category = ErrorClassifier.get_error_category(error)
    should_retry = ErrorClassifier.should_retry(error)
    is_fatal = ErrorClassifier.is_fatal(error)

    return category, should_retry, is_fatal


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 5.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retry_on: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for retrying functions with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for exponential backoff
        max_delay: Maximum delay between retries
        retry_on: List of exception types to retry on (None = use default)

    Returns:
        Decorated function with retry logic
    """
    if retry_on is None:
        retry_on = ErrorClassifier.RETRYABLE_ERRORS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this is the last attempt
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries")
                        raise

                    # Classify error
                    category, should_retry, is_fatal = classify_error(e)

                    # Don't retry fatal errors
                    if is_fatal:
                        logger.error(f"{func.__name__} failed with fatal error: {category}")
                        logger.error(f"Error details: {str(e)}")
                        raise

                    # Don't retry if not in retry list
                    if not should_retry:
                        logger.error(f"{func.__name__} failed with non-retryable error: {category}")
                        raise

                    # Log retry attempt
                    logger.warning(
                        f"{func.__name__} failed with {category} (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    logger.debug(f"Error details: {str(e)}")

                    # Wait before retry
                    time.sleep(delay)

                    # Calculate next delay with exponential backoff
                    delay = min(delay * backoff_factor, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        # Async version
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            import asyncio

            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if this is the last attempt
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries")
                        raise

                    # Classify error
                    category, should_retry, is_fatal = classify_error(e)

                    # Don't retry fatal errors
                    if is_fatal:
                        logger.error(f"{func.__name__} failed with fatal error: {category}")
                        logger.error(f"Error details: {str(e)}")
                        raise

                    # Don't retry if not in retry list
                    if not should_retry:
                        logger.error(f"{func.__name__} failed with non-retryable error: {category}")
                        raise

                    # Log retry attempt
                    logger.warning(
                        f"{func.__name__} failed with {category} (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    logger.debug(f"Error details: {str(e)}")

                    # Wait before retry
                    await asyncio.sleep(delay)

                    # Calculate next delay with exponential backoff
                    delay = min(delay * backoff_factor, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


def format_error_message(error: Exception) -> str:
    """
    Format error message with troubleshooting guidance

    Args:
        error: Exception to format

    Returns:
        Formatted error message with guidance
    """
    category, _, is_fatal = classify_error(error)

    message = f"❌ {category}: {str(error)}\n"

    if isinstance(error, AuthenticationError):
        message += (
            "\nPossible causes:\n"
            "1. Invalid API key in .env file\n"
            "2. API key has insufficient credits\n"
            "3. API key doesn't have access to the specified model\n"
            "\n"
            "Troubleshooting:\n"
            "1. Verify API key: https://openrouter.ai/settings/keys\n"
            "2. Check credits: https://openrouter.ai/credits\n"
            "3. Run: python generate_testset.py --validate-api\n"
        )

    elif isinstance(error, RateLimitError):
        message += (
            "\nYou've exceeded the rate limit.\n"
            "\n"
            "Actions:\n"
            "1. Wait a few minutes before retrying\n"
            "2. Consider upgrading your OpenRouter plan\n"
            "3. The system will automatically retry with backoff\n"
        )

    elif isinstance(error, NotFoundError):
        message += (
            "\nThe specified model was not found or is not accessible.\n"
            "\n"
            "Actions:\n"
            "1. Verify model name in config.yaml\n"
            "2. Check model availability: https://openrouter.ai/models\n"
            "3. Ensure your API key has access to the model\n"
        )

    elif isinstance(error, APIConnectionError):
        message += (
            "\nFailed to connect to the API.\n"
            "\n"
            "Actions:\n"
            "1. Check your internet connection\n"
            "2. Verify OpenRouter service status\n"
            "3. The system will automatically retry\n"
        )

    return message
