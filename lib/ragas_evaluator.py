"""
Ragas Evaluator Wrapper

Handles Ragas evaluation with custom LLM and embeddings configuration.
"""

import time
from typing import Any, Dict, List, Optional

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    AnswerSimilarity,
    ContextEntityRecall,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
    Faithfulness,
)

from .logger import get_logger


class RagasEvaluator:
    """Wrapper for Ragas evaluation with custom configuration"""

    # Available metrics
    AVAILABLE_METRICS = {
        'context_precision': ContextPrecision(),
        'context_recall': ContextRecall(),
        'context_entity_recall': ContextEntityRecall(),
        'answer_relevancy': AnswerRelevancy(),
        'faithfulness': Faithfulness(),
        'answer_correctness': AnswerCorrectness(),
        'answer_similarity': AnswerSimilarity(),
        'context_utilization': ContextUtilization(),
    }

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        llm_model: str = "x-ai/grok-code-fast-1",
        embedding_model: str = "qwen/qwen3-embedding-8b",
        llm_temperature: float = 0.7,
        timeout: int = 120,
        embeddings_timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize Ragas evaluator

        Args:
            api_key: OpenRouter API key
            api_base_url: OpenRouter API base URL
            llm_model: LLM model name for evaluation
            embedding_model: Embedding model name
            llm_temperature: Temperature for LLM
            timeout: Request timeout in seconds for LLM
            embeddings_timeout: Request timeout in seconds for embeddings
            max_retries: Maximum retry attempts
        """
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.llm_temperature = llm_temperature
        self.timeout = timeout
        self.embeddings_timeout = embeddings_timeout
        self.max_retries = max_retries

        # Initialize LLM and embeddings
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()

        self.logger.info(f"Initialized Ragas evaluator with LLM: {llm_model}")

    def _init_llm(self) -> ChatOpenAI:
        """
        Initialize LangChain LLM for Ragas

        Returns:
            ChatOpenAI instance configured for OpenRouter
        """
        try:
            llm = ChatOpenAI(
                model=self.llm_model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base_url,
                temperature=self.llm_temperature,
                timeout=self.timeout,
                request_timeout=self.timeout,
                max_retries=self.max_retries
            )
            self.logger.info(f"Initialized LLM: {self.llm_model} (timeout: {self.timeout}s)")
            return llm

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _init_embeddings(self) -> OpenAIEmbeddings:
        """
        Initialize LangChain embeddings for Ragas

        Returns:
            OpenAIEmbeddings instance configured for OpenRouter
        """
        try:
            embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base_url,
                timeout=self.embeddings_timeout,
                max_retries=self.max_retries
            )
            self.logger.info(f"Initialized embeddings: {self.embedding_model} (timeout: {self.embeddings_timeout}s)")
            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _reinitialize_models(self):
        """Reinitialize LLM and embeddings (used for retry logic)"""
        self.logger.info("Reinitializing LLM and embeddings for retry...")
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()

    def _convert_result_to_dict(self, result) -> Dict[str, Any]:
        """
        Convert EvaluationResult to dictionary format

        Args:
            result: EvaluationResult object from Ragas evaluate()

        Returns:
            Dictionary with aggregate metric scores
        """
        import numpy as np

        # EvaluationResult has a scores field which is List[Dict[str, Any]]
        # We need to compute aggregate scores (mean) for each metric

        scores_dict = {}

        # Get all metric names from the scores
        if hasattr(result, 'scores') and result.scores:
            # Collect all scores by metric name
            metric_scores = {}
            for sample_scores in result.scores:
                for metric_name, score_value in sample_scores.items():
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(score_value)

            # Compute mean for each metric
            for metric_name, scores_list in metric_scores.items():
                # Filter out None values
                valid_scores = [s for s in scores_list if s is not None and not (isinstance(s, float) and np.isnan(s))]
                if valid_scores:
                    scores_dict[metric_name] = np.mean(valid_scores)
                else:
                    scores_dict[metric_name] = 0.0

        self.logger.debug(f"Converted EvaluationResult to dict: {scores_dict}")
        return scores_dict

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> List[Any]:
        """
        Get list of Ragas metrics to evaluate

        Args:
            metric_names: List of metric names to use. If None, use all metrics.

        Returns:
            List of Ragas metric objects
        """
        if metric_names is None:
            # Use all available metrics
            metric_names = list(self.AVAILABLE_METRICS.keys())

        metrics = []
        for name in metric_names:
            if name in self.AVAILABLE_METRICS:
                metrics.append(self.AVAILABLE_METRICS[name])
                self.logger.debug(f"Added metric: {name}")
            else:
                self.logger.warning(f"Unknown metric: {name}")

        self.logger.info(f"Using {len(metrics)} metrics for evaluation")
        return metrics

    def evaluate(
        self,
        dataset: Dataset,
        metric_names: Optional[List[str]] = None,
        batch_size: int = 10,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run Ragas evaluation on dataset

        Args:
            dataset: Ragas Dataset to evaluate
            metric_names: List of metric names to use
            batch_size: Batch size for evaluation
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Ragas Evaluation")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset size: {len(dataset)}")
        self.logger.info(f"Batch size: {batch_size}")

        # Get metrics
        metrics = self.get_metrics(metric_names)
        self.logger.info(f"Metrics: {[m.name for m in metrics]}")

        try:
            start_time = time.time()

            # Run Ragas evaluation
            self.logger.info("\nRunning evaluation...")
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False  # Continue on errors
            )

            elapsed_time = time.time() - start_time

            self.logger.info(f"\n✓ Evaluation completed in {elapsed_time:.1f}s")
            self.logger.info(f"Average time per sample: {elapsed_time / len(dataset):.2f}s")

            # Convert EvaluationResult to dict format
            result_dict = self._convert_result_to_dict(result)

            # Log results summary
            self.logger.info("\nEvaluation Results:")
            for metric_name, score in result_dict.items():
                self.logger.info(f"  {metric_name}: {score:.4f}")

            # Return both the dict and the original result for detailed analysis
            return result_dict, result

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    def evaluate_with_retry(
        self,
        dataset: Dataset,
        metric_names: Optional[List[str]] = None,
        batch_size: int = 10,
        show_progress: bool = True,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> tuple:
        """
        Run Ragas evaluation with retry logic

        Args:
            dataset: Ragas Dataset to evaluate
            metric_names: List of metric names to use
            batch_size: Batch size for evaluation
            show_progress: Whether to show progress bar
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Tuple of (result_dict, evaluation_result)
        """
        for attempt in range(max_retries):
            try:
                return self.evaluate(
                    dataset=dataset,
                    metric_names=metric_names,
                    batch_size=batch_size,
                    show_progress=show_progress
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Evaluation failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    self.logger.info(f"Retrying in {retry_delay}s...")

                    # Reinitialize models before retry to ensure clean state
                    try:
                        self._reinitialize_models()
                    except Exception as reinit_error:
                        self.logger.error(f"Failed to reinitialize models: {reinit_error}")

                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Evaluation failed after {max_retries} attempts")
                    raise

    def evaluate_batched(
        self,
        dataset: Dataset,
        metric_names: Optional[List[str]] = None,
        batch_size: int = 10,
        checkpoint_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run Ragas evaluation in batches with checkpointing

        Args:
            dataset: Ragas Dataset to evaluate
            metric_names: List of metric names to use
            batch_size: Number of samples per batch
            checkpoint_callback: Optional callback function for checkpointing
                                 Called with (batch_num, batch_results)

        Returns:
            Dictionary with combined evaluation results
        """
        self.logger.info(f"Running batched evaluation (batch_size={batch_size})")

        total_samples = len(dataset)
        num_batches = (total_samples + batch_size - 1) // batch_size

        all_results = []
        metrics = self.get_metrics(metric_names)

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_samples)

            self.logger.info(
                f"\nProcessing batch {batch_num + 1}/{num_batches} "
                f"(samples {start_idx}-{end_idx})"
            )

            # Get batch
            batch_dataset = dataset.select(range(start_idx, end_idx))

            # Evaluate batch
            try:
                batch_result = self.evaluate(
                    dataset=batch_dataset,
                    metric_names=metric_names,
                    batch_size=len(batch_dataset),
                    show_progress=False
                )

                all_results.append(batch_result)

                # Checkpoint callback
                if checkpoint_callback:
                    checkpoint_callback(batch_num, batch_result)

            except Exception as e:
                self.logger.error(f"Batch {batch_num + 1} failed: {e}")
                # Continue with next batch

        # Combine results
        if not all_results:
            raise RuntimeError("All batches failed")

        combined_result = self._combine_batch_results(all_results, metrics)
        return combined_result

    def _combine_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        metrics: List[Any]
    ) -> Dict[str, Any]:
        """
        Combine results from multiple batches

        Args:
            batch_results: List of batch result dictionaries
            metrics: List of metrics used

        Returns:
            Combined result dictionary
        """
        # Simple averaging of metric scores
        combined = {}

        for metric in metrics:
            metric_name = metric.name
            scores = [
                result[metric_name]
                for result in batch_results
                if metric_name in result
            ]

            if scores:
                combined[metric_name] = sum(scores) / len(scores)

        self.logger.info("\nCombined batch results:")
        for metric_name, score in combined.items():
            self.logger.info(f"  {metric_name}: {score:.4f}")

        return combined
