#!/usr/bin/env python3
"""
RAG API Client for SmartKnowledge
Handles HTTP communication with the SmartKnowledge RAG application
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


class RAGClient:
    """HTTP client for SmartKnowledge RAG API"""

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        Initialize RAG API client

        Args:
            base_url: Base URL of SmartKnowledge API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    def health_check(self) -> bool:
        """
        Check if SmartKnowledge API is accessible

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try to access the API base URL
            # We just want to verify the server is responding
            response = self.session.get(
                f"{self.base_url}/",
                timeout=10
            )
            # Accept any response that means server is running
            # (including 404, which means server responded but no route at /)
            return True
        except requests.exceptions.RequestException:
            return False

    def call_chat_api(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call the chat-complete API with a query

        Args:
            query: User question to send to the RAG system
            session_id: Optional session ID (generates new one if not provided)

        Returns:
            Response dictionary with:
                - text: Generated answer
                - toolExecutions: List of tool calls made
                - status: "completed" or "error"
                - error: Error message if status is "error"

        Raises:
            requests.exceptions.RequestException: If API call fails after retries
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Construct request payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "sessionId": session_id,
            "sessionTitle": "RAG Evaluation",
            "sessionCreatedAt": int(time.time() * 1000)  # Current timestamp in ms
        }

        # Retry logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.session.post(
                    f"{self.base_url}/api/chat-complete",
                    json=payload,
                    timeout=self.timeout,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )

                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)

                # Check if request was successful
                response.raise_for_status()

                # Parse JSON response
                data = response.json()

                # Add latency to response
                data['latency_ms'] = latency_ms

                return data

            except requests.exceptions.RequestException as e:
                last_exception = e

                # Don't retry on client errors (4xx)
                if hasattr(e, 'response') and e.response is not None:
                    if 400 <= e.response.status_code < 500:
                        raise

                # Wait before retrying (except on last attempt)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        # All retries failed
        raise last_exception

    def parse_tool_executions(self, response: Dict[str, Any]) -> List[str]:
        """
        Extract retrieved contexts from tool executions

        Args:
            response: API response from call_chat_api

        Returns:
            List of retrieved context strings (chunk texts)
        """
        contexts = []

        # Get tool executions from response
        tool_executions = response.get('toolExecutions', [])

        # Filter for search-related tools
        search_tools = [
            'hybrid_search',
            'vector_search',
            'full_text_search',
            'get_contextual_chunks'
        ]

        for execution in tool_executions:
            tool_name = execution.get('tool', '')

            # Only process search tools
            if tool_name not in search_tools:
                continue

            # Get the result (should be JSON string)
            result_str = execution.get('result', '')

            if not result_str:
                continue

            try:
                # Parse the result JSON
                result_data = json.loads(result_str)

                # Result should be an array of ChunkResult objects
                if isinstance(result_data, list):
                    for chunk in result_data:
                        if isinstance(chunk, dict):
                            # Extract text_snippet from each chunk
                            text_snippet = chunk.get('text_snippet', '')
                            if text_snippet:
                                contexts.append(text_snippet)

            except json.JSONDecodeError:
                # Skip malformed JSON
                continue

        return contexts

    def get_answer(self, response: Dict[str, Any]) -> str:
        """
        Extract the generated answer from API response

        Args:
            response: API response from call_chat_api

        Returns:
            Generated answer text
        """
        return response.get('text', '')

    def get_tool_calls_json(self, response: Dict[str, Any]) -> str:
        """
        Get tool executions as JSON string for debugging

        Args:
            response: API response from call_chat_api

        Returns:
            JSON string of tool executions
        """
        tool_executions = response.get('toolExecutions', [])
        return json.dumps(tool_executions, ensure_ascii=False)

    def get_latency(self, response: Dict[str, Any]) -> int:
        """
        Get API latency in milliseconds

        Args:
            response: API response from call_chat_api

        Returns:
            Latency in milliseconds
        """
        return response.get('latency_ms', 0)

    def close(self):
        """Close the HTTP session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
