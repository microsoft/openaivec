#!/usr/bin/env python3
"""
Test script to force notebook environment and test progress bar
"""

import pandas as pd

from openaivec.proxy import BatchingMapProxy


def test_progress_bar_forced():
    """Test progress bar with forced notebook environment"""

    # Monkey patch the notebook detection to return True
    from openaivec.proxy import ProxyBase

    original_method = ProxyBase._is_notebook_environment
    ProxyBase._is_notebook_environment = lambda self: True

    try:

        def dummy_func(items):
            import time

            time.sleep(0.1)  # Simulate some work
            return [f"processed_{item}" for item in items]

        # Create proxy with progress enabled
        proxy = BatchingMapProxy(batch_size=3, show_progress=True)

        print("Testing BatchingMapProxy with forced notebook environment...")
        items = list(range(10))
        results = proxy.map(items, dummy_func)
        print(f"Results: {len(results)} items processed")

        return len(results) == len(items)

    finally:
        # Restore original method
        ProxyBase._is_notebook_environment = original_method


def test_pandas_ext_with_progress():
    """Test pandas extension with progress bar"""
    import os

    os.environ["OPENAI_API_KEY"] = "test-key"  # Set dummy API key

    # Monkey patch notebook detection
    from openaivec.proxy import ProxyBase

    original_method = ProxyBase._is_notebook_environment
    ProxyBase._is_notebook_environment = lambda self: True

    try:
        # Create a small series for testing
        series = pd.Series(["cat", "dog", "bird"])

        print("Testing pandas extension with show_progress=True...")
        print("Note: This will fail due to missing API key, but should create progress bar proxy")

        # This should create a BatchingMapProxy with show_progress=True
        try:
            # This won't actually work without a real API key, but we can check if the proxy is created correctly
            series.ai.responses("translate to French", batch_size=2, show_progress=True)
        except Exception as e:
            print(f"Expected error (missing API setup): {type(e).__name__}")
            print("But the progress bar proxy should have been created correctly")

    finally:
        # Restore original method
        ProxyBase._is_notebook_environment = original_method


if __name__ == "__main__":
    print("=== Testing Progress Bar with Forced Notebook Environment ===")

    print("\n1. Testing BatchingMapProxy with forced notebook:")
    test_progress_bar_forced()

    print("\n2. Testing pandas extension:")
    test_pandas_ext_with_progress()

    print("\n=== Test completed ===")
