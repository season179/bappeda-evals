"""
OpenRouter-compatible ChatOpenAI wrapper for Ragas evaluation.
"""
import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult, Generation
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class OpenRouterChatOpenAI(ChatOpenAI):
    """
    Custom ChatOpenAI wrapper for OpenRouter compatibility.

    OpenRouter doesn't support the 'n' parameter for multiple completions.
    This wrapper overrides agenerate_prompt to manually make separate API calls
    for each prompt, ensuring Ragas gets the expected number of generations.

    When Ragas needs multiple generations (e.g., strictness=3 in AnswerRelevancy),
    it will send 3 prompts, and this wrapper ensures each gets processed separately,
    returning 3 generation groups as expected.

    Example:
        >>> llm = OpenRouterChatOpenAI(
        ...     model="x-ai/grok-code-fast-1",
        ...     openai_api_key=api_key,
        ...     openai_api_base="https://openrouter.ai/api/v1"
        ... )
        >>> # Use with Ragas - it will automatically batch requests
        >>> evaluator = RagasEvaluator(llm=llm)
    """

    def __init__(self, **kwargs):
        """Initialize with n=1 to ensure single generation per request."""
        super().__init__(**kwargs)
        # Force n=1 to prevent OpenRouter from receiving unsupported parameter
        self.n = 1
        logger.debug(f"OpenRouterChatOpenAI initialized with n={self.n}")

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        Get default parameters, excluding 'n' for OpenRouter compatibility.

        Returns:
            Dictionary of parameters without the 'n' key
        """
        params = super()._default_params

        # Remove 'n' parameter as OpenRouter doesn't support it
        if 'n' in params:
            logger.debug(f"Removing 'n' parameter (was: {params['n']}) for OpenRouter compatibility")
            params.pop('n')

        return params

    async def agenerate_prompt(
        self,
        prompts: List[LanguageModelInput],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Override agenerate_prompt to make separate API calls for each prompt.

        This ensures OpenRouter processes each prompt independently, returning
        one generation per prompt as expected by Ragas.

        Args:
            prompts: List of prompts to generate from
            stop: Optional stop sequences
            callbacks: Optional callbacks
            **kwargs: Additional arguments

        Returns:
            LLMResult with one generation group per prompt
        """
        logger.debug(f"agenerate_prompt called with {len(prompts)} prompts")

        # Process each prompt separately to ensure OpenRouter returns one result per prompt
        all_generations: List[List[Generation]] = []
        combined_llm_output: Dict[str, Any] = {}

        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing prompt {i+1}/{len(prompts)}")

            # Call parent's agenerate_prompt with single prompt
            result = await super().agenerate_prompt(
                [prompt],  # Single prompt
                stop=stop,
                callbacks=callbacks,
                **kwargs
            )

            # Extract the first (and only) generation group
            if result.generations:
                all_generations.append(result.generations[0])
                logger.debug(f"  Got {len(result.generations[0])} generations for prompt {i+1}")
            else:
                logger.warning(f"  No generations returned for prompt {i+1}")
                # Add empty generation to maintain indexing
                all_generations.append([])

            # Accumulate token usage info
            if result.llm_output:
                for key, value in result.llm_output.items():
                    if key in combined_llm_output:
                        # Sum numeric values (like token counts)
                        if isinstance(value, (int, float)) and isinstance(combined_llm_output[key], (int, float)):
                            combined_llm_output[key] += value
                    else:
                        combined_llm_output[key] = value

        logger.debug(f"agenerate_prompt returning {len(all_generations)} generation groups")

        return LLMResult(
            generations=all_generations,
            llm_output=combined_llm_output if combined_llm_output else None,
        )
