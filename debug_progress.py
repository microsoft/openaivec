#!/usr/bin/env python3
"""
Debug script to test progress bar functionality
"""

from openaivec.proxy import BatchingMapProxy


def test_notebook_detection():
    """Test notebook environment detection"""
    from openaivec.proxy import ProxyBase

    proxy = ProxyBase()
    is_notebook = proxy._is_notebook_environment()
    print(f"Notebook environment detected: {is_notebook}")
    return is_notebook


def test_progress_bar_creation():
    """Test progress bar creation"""
    from openaivec.proxy import ProxyBase

    proxy = ProxyBase()
    proxy.show_progress = True

    progress_bar = proxy._create_progress_bar(100, "Testing")
    print(f"Progress bar created: {progress_bar is not None}")
    if progress_bar:
        progress_bar.close()
    return progress_bar is not None


def test_batching_proxy_with_progress():
    """Test BatchingMapProxy with progress"""

    def dummy_func(items):
        import time

        time.sleep(0.1)  # Simulate some work
        return [f"processed_{item}" for item in items]

    # Create proxy with progress enabled
    proxy = BatchingMapProxy(batch_size=5, show_progress=True)

    print("Testing BatchingMapProxy with progress...")
    items = list(range(20))
    results = proxy.map(items, dummy_func)
    print(f"Results: {len(results)} items processed")
    return len(results) == len(items)


if __name__ == "__main__":
    print("=== Debug Progress Bar Functionality ===")

    print("\n1. Testing notebook environment detection:")
    test_notebook_detection()

    print("\n2. Testing progress bar creation:")
    test_progress_bar_creation()

    print("\n3. Testing BatchingMapProxy with progress:")
    test_batching_proxy_with_progress()

    print("\n=== Debug completed ===")
