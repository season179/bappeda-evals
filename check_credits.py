#!/usr/bin/env python3
"""
OpenRouter Credit Balance Checker

A standalone script to check your OpenRouter account credit balance.
Requires OPENROUTER_API_KEY to be set in the .env file.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


def load_api_key() -> str:
    """Load OpenRouter API key from .env file."""
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        sys.exit(1)

    return api_key


def check_credits(api_key: str) -> dict:
    """
    Check OpenRouter credit balance.

    Args:
        api_key: OpenRouter API key

    Returns:
        dict: Credit information from the API

    Raises:
        httpx.HTTPError: If the API request fails
    """
    url = "https://openrouter.ai/api/v1/credits"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    with httpx.Client() as client:
        response = client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response.json()


def display_credits(credit_data: dict) -> None:
    """
    Display credit information in a user-friendly format.

    Args:
        credit_data: Credit information from the API
    """
    print("\n" + "="*50)
    print("OpenRouter Credit Balance")
    print("="*50)

    # Extract the actual data if it's wrapped in a 'data' field
    data = credit_data.get('data', credit_data)

    if isinstance(data, dict):
        # Calculate remaining balance
        total_credits = data.get('total_credits', 0)
        total_usage = data.get('total_usage', 0)
        remaining = total_credits - total_usage

        print(f"Total Credits: ${total_credits:.2f}")
        print(f"Total Usage: ${total_usage:.2f}")
        print(f"Remaining Balance: ${remaining:.2f}")

        # Display any other fields that might be present
        other_fields = {k: v for k, v in data.items()
                       if k not in ['total_credits', 'total_usage']}
        if other_fields:
            print("\nAdditional Information:")
            for key, value in other_fields.items():
                display_key = key.replace("_", " ").title()
                if isinstance(value, (int, float)):
                    print(f"  {display_key}: ${value:.2f}")
                else:
                    print(f"  {display_key}: {value}")
    else:
        print(f"Raw response: {credit_data}")

    print("="*50 + "\n")


def main():
    """Main function to check and display OpenRouter credits."""
    try:
        # Load API key
        api_key = load_api_key()

        # Check credits
        print("Checking OpenRouter credit balance...")
        credit_data = check_credits(api_key)

        # Display results
        display_credits(credit_data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print(f"\nError: Invalid API key (401 Unauthorized)")
            print("Please check your OPENROUTER_API_KEY in the .env file\n")
        elif e.response.status_code == 403:
            print(f"\nError: Access forbidden (403)")
            print("This API key may not have permission to check credits\n")
        else:
            print(f"\nError: API request failed with status {e.response.status_code}")
            print(f"Response: {e.response.text}\n")
        sys.exit(1)

    except httpx.RequestError as e:
        print(f"\nError: Network request failed")
        print(f"Details: {e}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: An unexpected error occurred")
        print(f"Details: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
