"""
Ragas Evaluator Wrapper

Handles Ragas evaluation with custom LLM and embeddings configuration.
"""

import time
from typing import Any, Dict, List, Optional

from datasets import Dataset
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
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
from .openrouter_chat import OpenRouterChatOpenAI


class RagasEvaluator:
    """Wrapper for Ragas evaluation with custom configuration"""

    # Available metric classes (not instances - we'll create fresh instances each time)
    AVAILABLE_METRIC_CLASSES = {
        'context_precision': ContextPrecision,
        'context_recall': ContextRecall,
        'context_entity_recall': ContextEntityRecall,
        'answer_relevancy': AnswerRelevancy,
        'faithfulness': Faithfulness,
        'answer_correctness': AnswerCorrectness,
        'answer_similarity': AnswerSimilarity,
        'context_utilization': ContextUtilization,
    }

    def __init__(
        self,
        api_key: str,
        api_base_url: str,
        llm_model: str,
        embedding_model: str,
        llm_temperature: float,
        timeout: int,
        embeddings_timeout: int,
        max_retries: int
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

    def _init_llm(self) -> LangchainLLMWrapper:
        """
        Initialize LangChain LLM for Ragas

        Uses OpenRouterChatOpenAI wrapper for compatibility with OpenRouter API,
        which doesn't support the 'n' parameter for multiple completions.
        Then wraps it in LangchainLLMWrapper with bypass_n=True to tell Ragas
        to use prompt batching instead of the 'n' parameter.

        Returns:
            LangchainLLMWrapper wrapping OpenRouterChatOpenAI
        """
        try:
            # Create base LangChain LLM
            base_llm = OpenRouterChatOpenAI(
                model=self.llm_model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base_url,
                temperature=self.llm_temperature,
                timeout=self.timeout,
                request_timeout=self.timeout,
                max_retries=self.max_retries
            )

            # Wrap in Ragas's LangchainLLMWrapper with bypass_n=True
            # This tells Ragas to NOT use the 'n' parameter, and instead
            # send multiple prompts: [prompt, prompt, prompt]
            llm = LangchainLLMWrapper(
                langchain_llm=base_llm,
                bypass_n=True  # Critical: tells Ragas to batch prompts instead of using n
            )

            self.logger.info(f"Initialized LLM: {self.llm_model} (timeout: {self.timeout}s)")
            self.logger.debug(f"Using OpenRouterChatOpenAI with bypass_n=True, wrapper class: {llm.__class__.__name__}")
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

        Creates fresh metric instances to avoid state issues between evaluations.

        Args:
            metric_names: List of metric names to use. If None, use all metrics.

        Returns:
            List of Ragas metric objects
        """
        if metric_names is None:
            # Use all available metrics
            metric_names = list(self.AVAILABLE_METRIC_CLASSES.keys())

        metrics = []
        for name in metric_names:
            if name in self.AVAILABLE_METRIC_CLASSES:
                try:
                    # Create fresh instance of the metric
                    metric_class = self.AVAILABLE_METRIC_CLASSES[name]
                    metric_instance = metric_class()
                    metrics.append(metric_instance)
                    self.logger.debug(f"Added metric: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize metric '{name}': {e}")
                    # Continue with other metrics instead of failing completely
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

    def evaluate_incremental(
        self,
        dataset: Dataset,
        metric_names: Optional[List[str]] = None,
        batch_size: int = 10,
        show_progress: bool = True,
        skip_sample_ids: Optional[List[int]] = None,
        retry_sample_ids: Optional[List[int]] = None,
        max_sample_attempts: int = 3
    ):
        """
        Run Ragas evaluation incrementally in batches, yielding results after each batch

        This enables:
        - Frequent disk persistence (after each batch)
        - Error resilience (continue on batch failures)
        - Resume capability (skip already-processed samples)

        Args:
            dataset: Ragas Dataset to evaluate
            metric_names: List of metric names to use
            batch_size: Number of samples per batch
            show_progress: Whether to show progress bar
            skip_sample_ids: Sample IDs to skip (already processed)
            retry_sample_ids: Sample IDs to retry (previously failed)
            max_sample_attempts: Maximum attempts per sample

        Yields:
            Tuple of (batch_results, batch_sample_ids, batch_status)
            - batch_results: List of dicts with metrics for each sample
            - batch_sample_ids: List of sample IDs in this batch
            - batch_status: "success" or "failed"
        """
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset

        skip_sample_ids = skip_sample_ids or []
        retry_sample_ids = retry_sample_ids or []

        self.logger.info("=" * 80)
        self.logger.info("Starting Incremental Ragas Evaluation")
        self.logger.info("=" * 80)
        self.logger.info(f"Dataset size: {len(dataset)}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Samples to skip: {len(skip_sample_ids)}")
        self.logger.info(f"Samples to retry: {len(retry_sample_ids)}")

        # Get metrics
        metrics = self.get_metrics(metric_names)
        self.logger.info(f"Metrics: {[m.name for m in metrics]}")

        # Convert dataset to list of samples for batch processing
        dataset_df = dataset.to_pandas()
        total_samples = len(dataset_df)

        # Determine which samples to process
        if retry_sample_ids:
            # Retry mode: only process failed samples
            samples_to_process = retry_sample_ids
            self.logger.info(f"Retry mode: processing {len(samples_to_process)} failed samples")
        else:
            # Normal mode: process all samples except skipped ones
            samples_to_process = [
                i for i in range(total_samples)
                if i not in skip_sample_ids
            ]
            self.logger.info(f"Processing {len(samples_to_process)} samples")

        # Process in batches
        num_batches = (len(samples_to_process) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(samples_to_process))
            batch_sample_ids = samples_to_process[batch_start:batch_end]

            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Batch {batch_idx + 1}/{num_batches} (samples {batch_sample_ids[0]}-{batch_sample_ids[-1]})")
            self.logger.info(f"{'='*80}")

            try:
                # Extract batch samples
                batch_df = dataset_df.iloc[batch_sample_ids]

                # Convert batch to Ragas Dataset
                batch_dataset = EvaluationDataset.from_pandas(batch_df)

                # Evaluate batch
                self.logger.info(f"Evaluating {len(batch_dataset)} samples...")
                start_time = time.time()

                result = evaluate(
                    dataset=batch_dataset,
                    metrics=metrics,
                    llm=self.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False  # Continue on errors within batch
                )

                elapsed_time = time.time() - start_time

                # Convert result to list of per-sample dicts
                result_df = result.dataset.to_pandas()
                batch_results = []

                for idx, row in result_df.iterrows():
                    sample_result = {
                        'user_input': row.get('user_input', ''),
                        'reference_contexts': row.get('reference_contexts', []),
                        'reference': row.get('reference', ''),
                        'retrieved_contexts': row.get('retrieved_contexts', []),
                        'response': row.get('response', '')
                    }

                    # Add metric scores
                    for metric in metrics:
                        metric_name = metric.name
                        sample_result[metric_name] = row.get(metric_name, 0.0)

                    batch_results.append(sample_result)

                self.logger.info(f"✓ Batch completed in {elapsed_time:.1f}s")
                self.logger.info(f"  Average: {elapsed_time / len(batch_dataset):.2f}s per sample")

                # Yield successful batch
                yield (batch_results, batch_sample_ids, "success")

            except Exception as e:
                self.logger.error(f"✗ Batch {batch_idx + 1} failed: {e}")
                self.logger.error(f"  Affected samples: {batch_sample_ids}")

                # Create empty results for failed batch
                failed_results = []
                for sample_id in batch_sample_ids:
                    row = dataset_df.iloc[sample_id]
                    failed_result = {
                        'user_input': row.get('user_input', ''),
                        'reference_contexts': row.get('reference_contexts', []),
                        'reference': row.get('reference', ''),
                        'retrieved_contexts': row.get('retrieved_contexts', []),
                        'response': ''
                    }
                    # Add zero scores for all metrics
                    for metric in metrics:
                        failed_result[metric.name] = 0.0

                    failed_results.append(failed_result)

                # Yield failed batch
                yield (failed_results, batch_sample_ids, "failed")

        self.logger.info(f"\n{'='*80}")
        self.logger.info("Incremental Evaluation Complete")
        self.logger.info(f"{'='*80}")
