"""
Error handling utilities for API calls

Provides error message formatting with troubleshooting guidance.
"""

from openai import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    APIConnectionError,
)


def format_error_message(error: Exception) -> str:
    """
    Format error message with troubleshooting guidance

    Args:
        error: Exception to format

    Returns:
        Formatted error message with guidance
    """
    message = f"{type(error).__name__}: {str(error)}\n"

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
            "3. Run: python generate_multihop_testset.py --validate-api\n"
        )

    elif isinstance(error, RateLimitError):
        message += (
            "\nYou've exceeded the rate limit.\n"
            "\n"
            "Actions:\n"
            "1. Wait a few minutes before retrying\n"
            "2. Consider upgrading your OpenRouter plan\n"
            "3. Reduce test_size in config.yaml\n"
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
            "3. Try again in a few moments\n"
        )

    return message
