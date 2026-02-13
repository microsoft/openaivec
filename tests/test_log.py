import asyncio
import json
import logging
import time

from openaivec._log import observe

EXPECTED_MIN_DURATION_NS = 5_000_000
EXPECTED_MAX_DURATION_NS = 1_000_000_000


def _create_logger(name: str) -> tuple[logging.Logger, list[str]]:
    logger = logging.getLogger(name)
    messages: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(_ListHandler())
    return logger, messages


class TestObserve:
    def test_function_without_args_is_logged(self):
        logger, messages = _create_logger("test.observe.function")

        @observe(logger)
        def run() -> int:
            return 1

        assert run() == 1

        assert len(messages) == 2
        start = json.loads(messages[0])
        end = json.loads(messages[1])
        assert start["class"] == "<function>"
        assert end["class"] == "<function>"

    def test_sync_function_logs_start_and_end(self):
        logger, messages = _create_logger("test.observe.sync")

        class Sample:
            @observe(logger)
            def run(self) -> int:
                time.sleep(0.01)
                return 1

        assert Sample().run() == 1

        assert len(messages) == 2
        start = json.loads(messages[0])
        end = json.loads(messages[1])
        assert start["type"] == "start"
        assert end["type"] == "end"
        assert start["transaction_id"] == end["transaction_id"]
        duration_ns = end["logged_at"] - start["logged_at"]
        assert duration_ns >= EXPECTED_MIN_DURATION_NS
        assert duration_ns < EXPECTED_MAX_DURATION_NS

    def test_async_function_logs_end_after_await(self):
        logger, messages = _create_logger("test.observe.async")

        class Sample:
            @observe(logger)
            async def run(self) -> int:
                await asyncio.sleep(0.02)
                return 1

        assert asyncio.run(Sample().run()) == 1

        assert len(messages) == 2
        start = json.loads(messages[0])
        end = json.loads(messages[1])
        assert start["type"] == "start"
        assert end["type"] == "end"
        assert start["transaction_id"] == end["transaction_id"]
        duration_ns = end["logged_at"] - start["logged_at"]
        assert duration_ns >= EXPECTED_MIN_DURATION_NS
        assert duration_ns < EXPECTED_MAX_DURATION_NS
