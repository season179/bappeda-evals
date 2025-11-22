"""
Report Generator for Ragas Evaluation

Generates human-readable markdown reports from Ragas evaluation results.
"""

import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .logger import get_logger


# Report generation constants
DEFAULT_BAR_WIDTH = 40
DEFAULT_TOP_N = 5
DEFAULT_TEXT_TRUNCATE_LENGTH = 100


class RagasReportGenerator:
    """Generates markdown reports from Ragas evaluation results"""

    def __init__(self):
        """Initialize the report generator"""
        self.logger = get_logger(__name__)

    def _has_metadata(self, detailed_results: pd.DataFrame) -> bool:
        """Check if DataFrame has _metadata column"""
        return '_metadata' in detailed_results.columns

    def _safe_get_metadata(self, row, key: str, default=None):
        """Safely get metadata value from row"""
        if not isinstance(row.get('_metadata'), dict):
            return default
        return row['_metadata'].get(key, default)

    def generate_report(
        self,
        results: Dict[str, Any],
        detailed_results: pd.DataFrame,
        output_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Generate comprehensive markdown report

        Args:
            results: Aggregate Ragas evaluation results
            detailed_results: DataFrame with per-query results
            output_path: Path to save report
            config: Optional configuration dictionary for context
        """
        self.logger.info(f"Generating report at {output_path}")

        report_lines = []

        # Header
        report_lines.extend(self._generate_header(config))

        # Executive Summary
        report_lines.extend(self._generate_executive_summary(results, detailed_results))

        # Metric Breakdown
        report_lines.extend(self._generate_metric_breakdown(results, detailed_results))

        # Best and Worst Queries
        report_lines.extend(self._generate_best_worst_queries(detailed_results))

        # Failure Analysis
        report_lines.extend(self._generate_failure_analysis(detailed_results))

        # Recommendations
        report_lines.extend(self._generate_recommendations(results, detailed_results))

        # Write report
        report_content = '\n'.join(report_lines)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"Report saved to {output_path}")

    def _generate_header(self, config: Optional[Dict[str, Any]]) -> List[str]:
        """Generate report header"""
        lines = [
            "# Ragas Evaluation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        if config:
            llm_model = config.get('llm_model', 'N/A')
            embedding_model = config.get('embedding_model', 'N/A')
            lines.extend([
                "**Configuration**:",
                f"- LLM Model: `{llm_model}`",
                f"- Embedding Model: `{embedding_model}`",
                ""
            ])

        lines.extend([
            "---",
            ""
        ])

        return lines

    def _generate_executive_summary(
        self,
        results: Dict[str, Any],
        detailed_results: pd.DataFrame
    ) -> List[str]:
        """Generate executive summary section"""
        lines = [
            "## Executive Summary",
            ""
        ]

        # Dataset statistics
        total_queries = len(detailed_results)

        # Safely count queries with contexts
        if self._has_metadata(detailed_results):
            queries_with_contexts = sum(
                detailed_results['_metadata'].apply(lambda x: x.get('has_contexts', False) if isinstance(x, dict) else False)
            )
        else:
            # Fallback: check retrieved_contexts column
            if 'retrieved_contexts' in detailed_results.columns:
                queries_with_contexts = sum(
                    detailed_results['retrieved_contexts'].apply(lambda x: isinstance(x, list) and len(x) > 0)
                )
            else:
                queries_with_contexts = 0

        queries_without_contexts = total_queries - queries_with_contexts

        lines.extend([
            "### Dataset Statistics",
            "",
            f"- **Total Queries**: {total_queries}",
            f"- **Queries with Contexts**: {queries_with_contexts} ({queries_with_contexts/total_queries*100:.1f}%)",
            f"- **Queries without Contexts**: {queries_without_contexts} ({queries_without_contexts/total_queries*100:.1f}%)",
            ""
        ])

        # Overall metrics
        lines.extend([
            "### Overall Metrics",
            "",
            "| Metric | Score |",
            "|--------|-------|"
        ])

        # Sort metrics by name for consistency
        sorted_metrics = [(k, v) for k, v in results.items() if k != 'per_sample']
        sorted_metrics.sort(key=lambda x: x[0])

        for metric_name, score in sorted_metrics:
            # Format metric name for display
            display_name = metric_name.replace('_', ' ').title()
            lines.append(f"| {display_name} | {score:.4f} |")

        lines.extend(["", ""])

        return lines

    def _generate_metric_breakdown(
        self,
        results: Dict[str, Any],
        detailed_results: pd.DataFrame
    ) -> List[str]:
        """Generate detailed metric breakdown"""
        lines = [
            "## Metric Breakdown",
            ""
        ]

        # Calculate statistics for each metric
        metric_columns = [col for col in detailed_results.columns
                         if col not in ['user_input', 'retrieved_contexts', 'response',
                                       'reference', 'reference_contexts', '_metadata']]

        for metric in metric_columns:
            if metric in detailed_results.columns:
                values = detailed_results[metric].dropna()

                if len(values) > 0:
                    display_name = metric.replace('_', ' ').title()
                    lines.extend([
                        f"### {display_name}",
                        "",
                        f"- **Mean**: {values.mean():.4f}",
                        f"- **Median**: {values.median():.4f}",
                        f"- **Std Dev**: {values.std():.4f}",
                        f"- **Min**: {values.min():.4f}",
                        f"- **Max**: {values.max():.4f}",
                        ""
                    ])

                    # Distribution histogram (text-based)
                    lines.extend(self._generate_text_histogram(values, display_name))

        return lines

    def _generate_text_histogram(
        self,
        values: pd.Series,
        metric_name: str,
        bins: int = 10
    ) -> List[str]:
        """Generate text-based histogram"""
        lines = [
            f"**Distribution**:",
            "",
            "```"
        ]

        # Create bins
        hist, bin_edges = pd.cut(values, bins=bins, retbins=True, duplicates='drop')
        counts = hist.value_counts().sort_index()

        # Find max count for scaling
        max_count = counts.max()
        bar_width = DEFAULT_BAR_WIDTH

        # Generate bars
        for interval, count in counts.items():
            bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = '█' * bar_length
            lines.append(f"{interval}: {bar} {count}")

        lines.extend([
            "```",
            ""
        ])

        return lines

    def _generate_best_worst_queries(
        self,
        detailed_results: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N
    ) -> List[str]:
        """Generate best and worst performing queries"""
        lines = [
            "## Best and Worst Performing Queries",
            ""
        ]

        # Use answer_correctness as the primary metric for ranking
        if 'answer_correctness' in detailed_results.columns:
            ranking_metric = 'answer_correctness'
        elif 'faithfulness' in detailed_results.columns:
            ranking_metric = 'faithfulness'
        else:
            # Use first available metric
            metric_columns = [col for col in detailed_results.columns
                            if col not in ['user_input', 'retrieved_contexts', 'response',
                                          'reference', 'reference_contexts', '_metadata']]
            if metric_columns:
                ranking_metric = metric_columns[0]
            else:
                lines.append("*No metrics available for ranking*\n")
                return lines

        # Sort by ranking metric
        sorted_df = detailed_results.sort_values(by=ranking_metric, ascending=False)

        # Best queries
        lines.extend([
            f"### Top {top_n} Best Queries",
            "",
            f"*Ranked by {ranking_metric.replace('_', ' ').title()}*",
            ""
        ])

        for i, (idx, row) in enumerate(sorted_df.head(top_n).iterrows(), 1):
            query_id = self._safe_get_metadata(row, 'query_id', idx)
            query = row['user_input'][:DEFAULT_TEXT_TRUNCATE_LENGTH] + "..." if len(row['user_input']) > DEFAULT_TEXT_TRUNCATE_LENGTH else row['user_input']
            score = row[ranking_metric]

            contexts_count = len(row['retrieved_contexts']) if isinstance(row.get('retrieved_contexts'), list) else 0

            lines.extend([
                f"**{i}. Query {query_id}** (Score: {score:.4f})",
                f"- Query: `{query}`",
                f"- Retrieved Contexts: {contexts_count}",
                ""
            ])

        # Worst queries
        lines.extend([
            f"### Top {top_n} Worst Queries",
            "",
            f"*Ranked by {ranking_metric.replace('_', ' ').title()}*",
            ""
        ])

        for i, (idx, row) in enumerate(sorted_df.tail(top_n).iloc[::-1].iterrows(), 1):
            query_id = self._safe_get_metadata(row, 'query_id', idx)
            query = row['user_input'][:DEFAULT_TEXT_TRUNCATE_LENGTH] + "..." if len(row['user_input']) > DEFAULT_TEXT_TRUNCATE_LENGTH else row['user_input']
            score = row[ranking_metric]
            error = self._safe_get_metadata(row, 'error', '')

            contexts_count = len(row['retrieved_contexts']) if isinstance(row.get('retrieved_contexts'), list) else 0

            lines.extend([
                f"**{i}. Query {query_id}** (Score: {score:.4f})",
                f"- Query: `{query}`",
                f"- Retrieved Contexts: {contexts_count}",
            ])

            if error:
                lines.append(f"- Error: {error}")

            lines.append("")

        return lines

    def _generate_failure_analysis(
        self,
        detailed_results: pd.DataFrame
    ) -> List[str]:
        """Generate failure analysis section"""
        lines = [
            "## Failure Analysis",
            ""
        ]

        # Count queries without contexts
        if self._has_metadata(detailed_results):
            no_contexts = sum(
                ~detailed_results['_metadata'].apply(lambda x: x.get('has_contexts', False) if isinstance(x, dict) else False)
            )
        else:
            # Fallback: check retrieved_contexts column
            if 'retrieved_contexts' in detailed_results.columns:
                no_contexts = sum(
                    ~detailed_results['retrieved_contexts'].apply(lambda x: isinstance(x, list) and len(x) > 0)
                )
            else:
                no_contexts = 0

        total_queries = len(detailed_results)

        lines.extend([
            f"### Queries Without Retrieved Contexts",
            "",
            f"- **Count**: {no_contexts} / {total_queries} ({no_contexts/total_queries*100:.1f}%)",
            ""
        ])

        if no_contexts > 0:
            lines.append("**Impact on Metrics**:")
            lines.append("")
            lines.append("Queries without contexts typically receive low or zero scores for:")
            lines.append("- Context Precision")
            lines.append("- Context Recall")
            lines.append("- Faithfulness")
            lines.append("- Answer Quality (due to lack of grounding)")
            lines.append("")

        # Error analysis
        if self._has_metadata(detailed_results):
            errors = detailed_results['_metadata'].apply(lambda x: x.get('error', '') if isinstance(x, dict) else '').value_counts()
            if len(errors) > 0 and errors.iloc[0] != '':
                lines.extend([
                    "### Error Summary",
                    "",
                    "| Error Type | Count |",
                    "|------------|-------|"
                ])

                for error, count in errors.items():
                    if error:  # Skip empty errors
                        error_short = error[:50] + "..." if len(error) > 50 else error
                        lines.append(f"| {error_short} | {count} |")

                lines.append("")

        # Latency analysis
        if self._has_metadata(detailed_results):
            latencies = detailed_results['_metadata'].apply(lambda x: x.get('api_latency_ms', 0) if isinstance(x, dict) else 0)
            if latencies.sum() > 0:
                lines.extend([
                    "### API Performance",
                    "",
                    f"- **Mean Latency**: {latencies.mean():.0f}ms",
                    f"- **Median Latency**: {latencies.median():.0f}ms",
                    f"- **P95 Latency**: {latencies.quantile(0.95):.0f}ms",
                    f"- **Max Latency**: {latencies.max():.0f}ms",
                    ""
                ])

        return lines

    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        detailed_results: pd.DataFrame
    ) -> List[str]:
        """Generate recommendations based on results"""
        lines = [
            "## Recommendations",
            ""
        ]

        recommendations = []

        # Check context recall
        if 'context_recall' in results:
            context_recall = results['context_recall']
            if context_recall < 0.5:
                recommendations.append(
                    "**Low Context Recall**: The retrieval system is missing important contexts. "
                    "Consider improving retrieval strategy, expanding the knowledge base, "
                    "or adjusting similarity thresholds."
                )

        # Check context precision
        if 'context_precision' in results:
            context_precision = results['context_precision']
            if context_precision < 0.5:
                recommendations.append(
                    "**Low Context Precision**: Retrieved contexts contain irrelevant information. "
                    "Consider improving ranking, filtering low-relevance results, "
                    "or using re-ranking techniques."
                )

        # Check faithfulness
        if 'faithfulness' in results:
            faithfulness = results['faithfulness']
            if faithfulness < 0.7:
                recommendations.append(
                    "**Low Faithfulness**: Answers may contain hallucinations. "
                    "Consider using more grounded prompts, improving context quality, "
                    "or implementing fact-checking mechanisms."
                )

        # Check answer correctness
        if 'answer_correctness' in results:
            answer_correctness = results['answer_correctness']
            if answer_correctness < 0.6:
                recommendations.append(
                    "**Low Answer Correctness**: Generated answers differ from ground truth. "
                    "Consider improving prompt engineering, using better LLM models, "
                    "or refining the knowledge base."
                )

        # Check for empty contexts
        if self._has_metadata(detailed_results):
            no_contexts = sum(
                ~detailed_results['_metadata'].apply(lambda x: x.get('has_contexts', False) if isinstance(x, dict) else False)
            )
        else:
            # Fallback: check retrieved_contexts column
            if 'retrieved_contexts' in detailed_results.columns:
                no_contexts = sum(
                    ~detailed_results['retrieved_contexts'].apply(lambda x: isinstance(x, list) and len(x) > 0)
                )
            else:
                no_contexts = 0

        if no_contexts > len(detailed_results) * 0.1:  # More than 10%
            recommendations.append(
                f"**High Context Failure Rate** ({no_contexts} queries): "
                "Many queries are not retrieving contexts. Investigate search functionality, "
                "index health, and query processing."
            )

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
                lines.append("")
        else:
            lines.append("*No critical issues identified. System performance is satisfactory.*")
            lines.append("")

        lines.extend([
            "---",
            "",
            "*Report generated by Ragas Evaluation System*"
        ])

        return lines
