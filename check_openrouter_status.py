#!/usr/bin/env python3
"""
OpenRouter Account Status Checker

A standalone script to check your OpenRouter account status, including:
- Credit balance and limits
- Rate limit information
- Usage statistics (daily, weekly, monthly)
- BYOK (Bring Your Own Key) usage
- Account tier status

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
    Get remaining credits for the authenticated user.

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


def check_status(api_key: str) -> dict:
    """
    Check OpenRouter account status, including credits and rate limits.

    Args:
        api_key: OpenRouter API key

    Returns:
        dict: Account status information from the API

    Raises:
        httpx.HTTPError: If the API request fails
    """
    url = "https://openrouter.ai/api/v1/key"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    with httpx.Client() as client:
        response = client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response.json()


def display_credits(credit_data: dict) -> None:
    """
    Display remaining credits information.

    Args:
        credit_data: Credit information from the API
    """
    print("\n" + "="*60)
    print("OpenRouter Credits")
    print("="*60)

    # Extract the actual data if it's wrapped in a 'data' field
    data = credit_data.get('data', credit_data)

    if not isinstance(data, dict):
        print(f"Raw response: {credit_data}")
        print("="*60 + "\n")
        return

    # Display credit information
    print("\n💰 Credit Summary")
    print("-" * 60)

    total_credits = data.get('total_credits', 0)
    total_usage = data.get('total_usage', 0)
    remaining = total_credits - total_usage

    print(f"Total Credits Purchased: ${total_credits:.2f}")
    print(f"Total Usage: ${total_usage:.2f}")
    print(f"Remaining Balance: ${remaining:.2f}")

    print("\n" + "="*60 + "\n")


def display_status(status_data: dict) -> None:
    """
    Display OpenRouter account status in a user-friendly format.

    Args:
        status_data: Account status information from the API
    """
    print("\n" + "="*60)
    print("OpenRouter Account Status")
    print("="*60)

    # Extract the actual data if it's wrapped in a 'data' field
    data = status_data.get('data', status_data)

    if not isinstance(data, dict):
        print(f"Raw response: {status_data}")
        print("="*60 + "\n")
        return

    # Account Overview
    print("\n📋 Account Overview")
    print("-" * 60)
    if data.get('label'):
        print(f"API Key Label: {data['label']}")

    tier_status = "Free Tier" if data.get('is_free_tier', False) else "Paid Account"
    print(f"Account Type: {tier_status}")

    # Credit Limits & Balance
    print("\n💳 Credit Limits & Balance")
    print("-" * 60)

    limit = data.get('limit')
    usage = data.get('usage', 0)
    limit_remaining = data.get('limit_remaining')

    # Show credit cap
    if limit is None:
        print("Credit Cap: Unlimited")
    else:
        print(f"Credit Cap: ${limit:.2f}")

    # Show total usage
    print(f"Total Usage: ${usage:.2f}")

    # Show remaining balance
    if limit_remaining is None:
        if limit is None:
            print("Remaining Balance: Unlimited")
        else:
            # Calculate remaining if limit exists but limit_remaining is not provided
            remaining = limit - usage
            print(f"Remaining Balance: ${remaining:.2f}")
    else:
        print(f"Remaining Balance: ${limit_remaining:.2f}")

    # Show reset schedule
    limit_reset = data.get('limit_reset')
    if limit_reset:
        print(f"Limit Reset Schedule: {limit_reset}")
    else:
        print("Limit Reset: Never")

    # Usage Statistics
    print("\n📊 Usage Statistics")
    print("-" * 60)

    usage = data.get('usage', 0)
    print(f"All-Time Usage: ${usage:.2f}")

    # Time-scoped usage
    if data.get('usage_daily') is not None:
        print(f"Daily Usage: ${data.get('usage_daily', 0):.2f}")
    if data.get('usage_weekly') is not None:
        print(f"Weekly Usage: ${data.get('usage_weekly', 0):.2f}")
    if data.get('usage_monthly') is not None:
        print(f"Monthly Usage: ${data.get('usage_monthly', 0):.2f}")

    # BYOK (Bring Your Own Key) Information
    byok_usage = data.get('byok_usage', 0)
    if byok_usage > 0 or any(data.get(k) for k in ['byok_usage_daily', 'byok_usage_weekly', 'byok_usage_monthly']):
        print("\n🔑 BYOK (Bring Your Own Key) Usage")
        print("-" * 60)
        print(f"All-Time BYOK Usage: ${byok_usage:.2f}")

        if data.get('byok_usage_daily') is not None:
            print(f"Daily BYOK Usage: ${data.get('byok_usage_daily', 0):.2f}")
        if data.get('byok_usage_weekly') is not None:
            print(f"Weekly BYOK Usage: ${data.get('byok_usage_weekly', 0):.2f}")
        if data.get('byok_usage_monthly') is not None:
            print(f"Monthly BYOK Usage: ${data.get('byok_usage_monthly', 0):.2f}")

        include_byok = data.get('include_byok_in_limit', False)
        print(f"Include BYOK in Limit: {'Yes' if include_byok else 'No'}")

    print("\n" + "="*60 + "\n")


def main():
    """Main function to check and display OpenRouter account status."""
    try:
        # Load API key
        api_key = load_api_key()

        print("Checking OpenRouter account status...")

        # Check credits
        try:
            credit_data = check_credits(api_key)
            display_credits(credit_data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                # Provisioning key required for credits endpoint
                print("\n⚠️  Credits endpoint requires provisioning key (skipping)")
                print("    Using account status endpoint instead...\n")
            else:
                raise

        # Check detailed status
        status_data = check_status(api_key)
        display_status(status_data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print(f"\nError: Invalid API key (401 Unauthorized)")
            print("Please check your OPENROUTER_API_KEY in the .env file\n")
        elif e.response.status_code == 403:
            print(f"\nError: Access forbidden (403)")
            print("This API key may not have permission to check account status\n")
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
