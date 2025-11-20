"""
API validation and pre-flight checks for OpenRouter

Tests connectivity and authentication before starting long-running operations.
"""

import asyncio
from typing import Dict, Tuple

from langchain_openai import ChatOpenAI
from openai import OpenAI, AuthenticationError, NotFoundError, RateLimitError

from .logger import get_logger


class APIValidator:
    """Validates API connectivity and credentials"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        llm_model: str,
        embedding_model: str
    ):
        """
        Initialize API validator

        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter base URL
            llm_model: LLM model to test
            embedding_model: Embedding model to test
        """
        self.api_key = api_key
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.logger = get_logger(__name__)

    def validate_all(self) -> Tuple[bool, Dict[str, str]]:
        """
        Run all validation checks

        Returns:
            Tuple of (success: bool, errors: Dict[str, str])
        """
        errors = {}

        self.logger.info("Running API pre-flight validation...")

        # Test LLM
        llm_success, llm_error = self._test_llm()
        if not llm_success:
            errors['llm'] = llm_error

        # Test Embeddings
        emb_success, emb_error = self._test_embeddings()
        if not emb_success:
            errors['embeddings'] = emb_error

        if errors:
            self.logger.error("API validation failed!")
            for service, error in errors.items():
                self.logger.error(f"  {service}: {error}")
            return False, errors

        self.logger.info("✅ API validation successful!")
        return True, {}

    def _test_llm(self) -> Tuple[bool, str]:
        """
        Test LLM API connectivity

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            self.logger.info(f"Testing LLM: {self.llm_model}...")

            llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.7,
                max_tokens=50  # Small request
            )

            # Test with simple prompt
            response = llm.invoke("Say 'test'")

            if response and response.content:
                self.logger.info(f"  ✅ LLM test successful")
                return True, ""

            self.logger.warning("  ⚠️  LLM returned empty response")
            return False, "LLM returned empty response"

        except AuthenticationError as e:
            error_msg = self._format_auth_error(str(e))
            self.logger.error(f"  ❌ Authentication failed: {error_msg}")
            return False, error_msg

        except NotFoundError as e:
            error_msg = f"Model '{self.llm_model}' not found or not accessible"
            self.logger.error(f"  ❌ {error_msg}")
            return False, error_msg

        except RateLimitError as e:
            error_msg = "Rate limit exceeded. Wait before retrying."
            self.logger.error(f"  ❌ {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"LLM test failed: {str(e)}"
            self.logger.error(f"  ❌ {error_msg}")
            return False, error_msg

    def _test_embeddings(self) -> Tuple[bool, str]:
        """
        Test Embeddings API connectivity

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            self.logger.info(f"Testing Embeddings: {self.embedding_model}...")

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            # Test with simple text
            response = client.embeddings.create(
                model=self.embedding_model,
                input="test"
            )

            if response and response.data and len(response.data) > 0:
                embedding_dim = len(response.data[0].embedding)
                self.logger.info(f"  ✅ Embeddings test successful (dimension: {embedding_dim})")
                return True, ""

            self.logger.warning("  ⚠️  Embeddings returned empty response")
            return False, "Embeddings returned empty response"

        except AuthenticationError as e:
            error_msg = self._format_auth_error(str(e))
            self.logger.error(f"  ❌ Authentication failed: {error_msg}")
            return False, error_msg

        except NotFoundError as e:
            error_msg = f"Model '{self.embedding_model}' not found or not accessible"
            self.logger.error(f"  ❌ {error_msg}")
            return False, error_msg

        except RateLimitError as e:
            error_msg = "Rate limit exceeded. Wait before retrying."
            self.logger.error(f"  ❌ {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"Embeddings test failed: {str(e)}"
            self.logger.error(f"  ❌ {error_msg}")
            return False, error_msg

    def _format_auth_error(self, error_str: str) -> str:
        """
        Format authentication error with helpful guidance

        Args:
            error_str: Original error string

        Returns:
            Formatted error message with troubleshooting steps
        """
        if "401" in error_str or "User not found" in error_str:
            return (
                "Authentication failed (401 - User not found)\n"
                "\n"
                "Possible causes:\n"
                "1. Invalid API key in .env file\n"
                "2. API key has insufficient credits\n"
                "3. API key doesn't have access to the specified model\n"
                "\n"
                "Troubleshooting:\n"
                "1. Verify API key: https://openrouter.ai/settings/keys\n"
                "2. Check credits: https://openrouter.ai/credits\n"
                "3. Verify model access in your account settings"
            )

        return error_str

    def get_validation_summary(self) -> str:
        """
        Get validation configuration summary

        Returns:
            Formatted summary string
        """
        lines = [
            "API Validation Configuration:",
            f"  Base URL: {self.base_url}",
            f"  LLM Model: {self.llm_model}",
            f"  Embedding Model: {self.embedding_model}",
            f"  API Key: {'*' * 20}{self.api_key[-8:] if len(self.api_key) >= 8 else '****'}"
        ]
        return "\n".join(lines)
