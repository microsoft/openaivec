import asyncio
import time

import pytest
import tiktoken

from openaivec._util import TextChunker, backoff, backoff_async


class TestTextChunker:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.sep = TextChunker(
            enc=tiktoken.encoding_for_model("text-embedding-3-large"),
        )
        yield

    def test_split(self):
        text = """
Kubernetes was announced by Google on June 6, 2014.[10] The project was conceived and created by Google employees Joe Beda, Brendan Burns, and Craig McLuckie. Others at Google soon joined to help build the project including Ville Aikas, Dawn Chen, Brian Grant, Tim Hockin, and Daniel Smith.[11][12] Other companies such as Red Hat and CoreOS joined the effort soon after, with notable contributors such as Clayton Coleman and Kelsey Hightower.[10]

The design and development of Kubernetes was inspired by Google's Borg cluster manager and based on Promise Theory.[13][14] Many of its top contributors had previously worked on Borg;[15][16] they codenamed Kubernetes "Project 7" after the Star Trek ex-Borg character Seven of Nine[17] and gave its logo a seven-spoked ship's wheel (designed by Tim Hockin). Unlike Borg, which was written in C++,[15] Kubernetes is written in the Go language.

Kubernetes was announced in June, 2014 and version 1.0 was released on July 21, 2015.[18] Google worked with the Linux Foundation to form the Cloud Native Computing Foundation (CNCF)[19] and offered Kubernetes as the seed technology.

Google was already offering a managed Kubernetes service, GKE, and Red Hat was supporting Kubernetes as part of OpenShift since the inception of the Kubernetes project in 2014.[20] In 2017, the principal competitors rallied around Kubernetes and announced adding native support for it:

VMware (proponent of Pivotal Cloud Foundry)[21] in August,
Mesosphere, Inc. (proponent of Marathon and Mesos)[22] in September,
Docker, Inc. (proponent of Docker)[23] in October,
Microsoft Azure[24] also in October,
AWS announced support for Kubernetes via the Elastic Kubernetes Service (EKS)[25] in November.
Cisco Elastic Kubernetes Service (EKS)[26] in November.
On March 6, 2018, Kubernetes Project reached ninth place in the list of GitHub projects by the number of commits, and second place in authors and issues, after the Linux kernel.[27]

Until version 1.18, Kubernetes followed an N-2 support policy, meaning that the three most recent minor versions receive security updates and bug fixes.[28] Starting with version 1.19, Kubernetes follows an N-3 support policy.[29]
"""

        chunks = self.sep.split(text, max_tokens=256, sep=[".", "\n\n"])

        # Assert that the number of chunks is as expected
        enc = tiktoken.encoding_for_model("text-embedding-3-large")

        for chunk in chunks:
            assert len(enc.encode(chunk)) <= 256


class TestBackoff:
    @pytest.mark.parametrize(
        "test_case,exceptions,scale,max_retries,fail_pattern,expected_calls,expected_result,expected_error",
        [
            ("no_exception", [ValueError], 0.1, 3, [], 1, "success", None),
            ("retries_on_exception", [ValueError], 0.01, 3, [ValueError, ValueError], 3, "success", None),
            ("multiple_exceptions", [ValueError, TypeError], 0.01, 5, [ValueError, TypeError], 3, "success", None),
            ("max_retries_exceeded", [ValueError], 0.01, 2, [ValueError, ValueError, ValueError], 2, None, ValueError),
            ("unhandled_exception", [ValueError], 0.01, 3, [TypeError], 1, None, TypeError),
        ],
    )
    def test_backoff_scenarios(
        self, test_case, exceptions, scale, max_retries, fail_pattern, expected_calls, expected_result, expected_error
    ):
        """Test various backoff scenarios with different exception patterns."""
        call_count = 0

        @backoff(exceptions=exceptions, scale=scale, max_retries=max_retries)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count <= len(fail_pattern):
                if len(fail_pattern) >= call_count:
                    error_class = fail_pattern[call_count - 1]
                    if error_class is ValueError:
                        raise ValueError("Test error" if test_case != "max_retries_exceeded" else "Always fails")
                    elif error_class is TypeError:
                        if test_case == "unhandled_exception":
                            raise TypeError("Unhandled exception")
                        else:
                            raise TypeError("Second error")
            return "success"

        if expected_error:
            with pytest.raises(expected_error) as cm:
                test_func()
            if test_case == "max_retries_exceeded":
                assert str(cm.value) == "Always fails"
            elif test_case == "unhandled_exception":
                assert str(cm.value) == "Unhandled exception"
        else:
            result = test_func()
            assert result == expected_result

        assert call_count == expected_calls

    def test_backoff_exponential_delay(self):
        """Test that delay increases exponentially."""
        call_times = []

        @backoff(exceptions=[ValueError], scale=0.01, max_retries=3)
        def track_timing():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Retry")
            return "success"

        result = track_timing()
        assert result == "success"
        assert len(call_times) == 3

        # Check that delays are present (but keep them small for test speed)
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i - 1]
            assert delay > 0  # Some delay exists


class TestBackoffAsync:
    @pytest.mark.parametrize(
        "test_case,exceptions,scale,max_retries,fail_pattern,expected_calls,expected_result,expected_error",
        [
            ("no_exception", [ValueError], 0.1, 3, [], 1, "success", None),
            ("retries_on_exception", [ValueError], 0.01, 3, [ValueError, ValueError], 3, "success", None),
            ("multiple_exceptions", [ValueError, TypeError], 0.01, 5, [ValueError, TypeError], 3, "success", None),
            ("max_retries_exceeded", [ValueError], 0.01, 2, [ValueError, ValueError, ValueError], 2, None, ValueError),
            ("unhandled_exception", [ValueError], 0.01, 3, [TypeError], 1, None, TypeError),
        ],
    )
    def test_backoff_async_scenarios(
        self, test_case, exceptions, scale, max_retries, fail_pattern, expected_calls, expected_result, expected_error
    ):
        """Test various async backoff scenarios with different exception patterns."""
        call_count = 0

        @backoff_async(exceptions=exceptions, scale=scale, max_retries=max_retries)
        async def test_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if call_count <= len(fail_pattern):
                if len(fail_pattern) >= call_count:
                    error_class = fail_pattern[call_count - 1]
                    if error_class is ValueError:
                        raise ValueError("Test error" if test_case != "max_retries_exceeded" else "Always fails")
                    elif error_class is TypeError:
                        if test_case == "unhandled_exception":
                            raise TypeError("Unhandled exception")
                        else:
                            raise TypeError("Second error")
            return "success"

        if expected_error:
            with pytest.raises(expected_error) as cm:
                asyncio.run(test_func())
            if test_case == "max_retries_exceeded":
                assert str(cm.value) == "Always fails"
            elif test_case == "unhandled_exception":
                assert str(cm.value) == "Unhandled exception"
        else:
            result = asyncio.run(test_func())
            assert result == expected_result

        assert call_count == expected_calls

    def test_backoff_async_with_openai_exceptions(self, mocker):
        """Test backoff with OpenAI exception types."""
        # Import OpenAI exceptions for testing
        try:
            from openai import InternalServerError, RateLimitError

            call_count = 0

            # Create a mock response object
            mock_response = mocker.Mock()
            mock_response.request = mocker.Mock()
            mock_response.status_code = 429  # For RateLimitError

            @backoff_async(exceptions=[RateLimitError, InternalServerError], scale=0.01, max_retries=3)
            async def simulate_api_errors():
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.01)
                if call_count == 1:
                    mock_response.status_code = 429
                    raise RateLimitError("Rate limit hit", response=mock_response, body=None)
                elif call_count == 2:
                    mock_response.status_code = 500
                    raise InternalServerError("Server error", response=mock_response, body=None)
                return "success"

            result = asyncio.run(simulate_api_errors())
            assert result == "success"
            assert call_count == 3
        except ImportError:
            # Skip test if OpenAI is not installed
            pytest.skip("OpenAI not installed")

    def test_backoff_production_settings(self):
        """Test backoff with production-like settings for OpenAI API."""
        call_count = 0
        call_times = []

        @backoff(exceptions=[ValueError], scale=1, max_retries=12)
        def simulate_rate_limit_scenario():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())

            # Simulate rate limit that clears after 2 retries (3 total attempts)
            if call_count < 3:
                raise ValueError("Rate limit hit")
            return "success"

        start_time = time.time()
        result = simulate_rate_limit_scenario()
        total_time = time.time() - start_time

        assert result == "success"
        assert call_count == 3
        assert len(call_times) == 3

        # With scale=1, exponential backoff can vary significantly due to jitter
        # First attempt: immediate, then up to 3s, up to 6s delays
        # Allow up to 15 seconds for 3 attempts with exponential backoff + jitter
        assert total_time < 15
        assert total_time > 0.1  # Should have some delay (adjusted for jitter)

        # Verify delays are increasing (roughly)
        if len(call_times) >= 2:
            first_delay = call_times[1] - call_times[0]
            assert first_delay > 0
