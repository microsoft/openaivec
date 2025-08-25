import time
from datetime import datetime, timezone
from threading import Thread

import pytest

from openaivec._optimize import BatchSizeSuggester, PerformanceMetric


class TestPerformanceMetric:
    def test_create_metric_without_exception(self):
        now = datetime.now(timezone.utc)
        metric = PerformanceMetric(duration=1.5, batch_size=10, executed_at=now)

        assert metric.duration == 1.5
        assert metric.batch_size == 10
        assert metric.executed_at == now
        assert metric.exception is None

    def test_create_metric_with_exception(self):
        now = datetime.now(timezone.utc)
        error = ValueError("test error")
        metric = PerformanceMetric(duration=1.5, batch_size=10, executed_at=now, exception=error)

        assert metric.exception == error


class TestBatchSizeSuggester:
    def test_default_initialization(self):
        suggester = BatchSizeSuggester()

        assert suggester.current_batch_size == 10
        assert suggester.min_batch_size == 10
        assert suggester.min_duration == 30.0
        assert suggester.max_duration == 60.0
        assert suggester.step_ratio == 0.2
        assert suggester.sample_size == 4
        assert len(suggester._history) == 0
        assert suggester._batch_size_changed_at is None

    def test_custom_initialization(self):
        suggester = BatchSizeSuggester(
            current_batch_size=20, min_batch_size=5, min_duration=15.0, max_duration=45.0, step_ratio=0.2, sample_size=5
        )

        assert suggester.current_batch_size == 20
        assert suggester.min_batch_size == 5
        assert suggester.min_duration == 15.0
        assert suggester.max_duration == 45.0
        assert suggester.step_ratio == 0.2
        assert suggester.sample_size == 5

    @pytest.mark.parametrize(
        "kwargs,expected_match",
        [
            ({"min_batch_size": 0}, "min_batch_size must be > 0"),
            ({"current_batch_size": 5, "min_batch_size": 10}, "current_batch_size must be >= min_batch_size"),
            ({"sample_size": 0}, "sample_size must be > 0"),
            ({"step_ratio": 0}, "step_ratio must be > 0"),
            ({"min_duration": 0}, "min_duration and max_duration must be > 0"),
            ({"max_duration": 0}, "min_duration and max_duration must be > 0"),
            ({"min_duration": 60, "max_duration": 30}, "min_duration must be < max_duration"),
        ],
    )
    def test_validation_errors(self, kwargs, expected_match):
        """Test various validation error scenarios."""
        with pytest.raises(ValueError, match=expected_match):
            BatchSizeSuggester(**kwargs)

    def test_record_success(self):
        suggester = BatchSizeSuggester()

        with suggester.record(batch_size=10):
            time.sleep(0.01)

        assert len(suggester._history) == 1
        metric = suggester._history[0]
        assert metric.batch_size == 10
        assert metric.duration > 0
        assert metric.exception is None

    def test_record_with_exception(self):
        suggester = BatchSizeSuggester()

        test_error = ValueError("test error")
        with pytest.raises(ValueError):
            with suggester.record(batch_size=10):
                raise test_error

        assert len(suggester._history) == 1
        metric = suggester._history[0]
        assert metric.batch_size == 10
        assert metric.exception == test_error

    def test_clear_history(self):
        suggester = BatchSizeSuggester()

        with suggester.record(batch_size=10):
            pass

        assert len(suggester._history) == 1
        suggester.clear_history()
        assert len(suggester._history) == 0

    def test_samples_empty_history(self):
        suggester = BatchSizeSuggester()
        assert suggester.samples == []

    def test_samples_with_valid_metrics(self):
        suggester = BatchSizeSuggester(sample_size=3)

        # Add some valid metrics
        for i in range(5):
            with suggester.record(batch_size=10):
                time.sleep(0.001)

        samples = suggester.samples
        assert len(samples) == 3  # Limited by sample_size

        # Should be in chronological order (oldest first)
        for i in range(len(samples) - 1):
            assert samples[i].executed_at <= samples[i + 1].executed_at

    def test_samples_excludes_exceptions(self):
        suggester = BatchSizeSuggester(sample_size=5)

        # Add valid metric
        with suggester.record(batch_size=10):
            time.sleep(0.001)

        # Add metric with exception
        with pytest.raises(ValueError):
            with suggester.record(batch_size=10):
                raise ValueError("test")

        # Add another valid metric
        with suggester.record(batch_size=10):
            time.sleep(0.001)

        samples = suggester.samples
        assert len(samples) == 2  # Only valid metrics
        assert all(m.exception is None for m in samples)

    def test_samples_after_batch_size_change(self):
        suggester = BatchSizeSuggester(sample_size=5, min_duration=1.0, max_duration=2.0)

        # Add some metrics
        for i in range(3):
            with suggester.record(batch_size=10):
                time.sleep(0.001)

        # Simulate batch size change
        change_time = datetime.now(timezone.utc)
        suggester._batch_size_changed_at = change_time

        # Add metrics after change
        for i in range(2):
            with suggester.record(batch_size=20):
                time.sleep(0.001)

        samples = suggester.samples
        # Should only include metrics after batch size change
        assert len(samples) == 2
        assert all(m.batch_size == 20 for m in samples)

    def test_suggest_batch_size_insufficient_samples(self):
        suggester = BatchSizeSuggester(sample_size=5)

        # Add only 2 metrics (less than sample_size)
        for i in range(2):
            with suggester.record(batch_size=10):
                time.sleep(0.001)

        assert suggester.suggest_batch_size() == 10  # No change

    @pytest.mark.parametrize(
        "scenario,sleep_duration,expected_size,should_change",
        [
            ("increase_when_too_fast", 0.1, 11, True),  # 0.1s < 0.5s min_duration
            ("decrease_when_too_slow", 1.5, 9, True),  # 1.5s > 1.0s max_duration
            ("no_change_in_range", 0.75, 10, False),  # 0.75s in range [0.5, 1.0]
        ],
    )
    def test_suggest_batch_size_scenarios(self, scenario, sleep_duration, expected_size, should_change):
        """Test batch size suggestion in different duration scenarios."""
        min_batch_size = 5 if scenario == "decrease_when_too_slow" else 10

        suggester = BatchSizeSuggester(
            current_batch_size=10,
            min_batch_size=min_batch_size,
            min_duration=0.5,  # 0.5 seconds (test scale)
            max_duration=1.0,  # 1.0 seconds (test scale)
            step_ratio=0.1,
            sample_size=3,
        )

        # Add metrics with specified duration
        for i in range(3):
            with suggester.record(batch_size=10):
                time.sleep(sleep_duration)

        new_size = suggester.suggest_batch_size()
        assert new_size == expected_size
        assert suggester.current_batch_size == expected_size

        if should_change:
            assert suggester._batch_size_changed_at is not None
        else:
            assert suggester._batch_size_changed_at is None

    def test_suggest_batch_size_respects_minimum(self):
        suggester = BatchSizeSuggester(
            current_batch_size=5,
            min_batch_size=5,
            min_duration=0.5,  # 0.5 seconds (test scale)
            max_duration=1.0,  # 1.0 seconds (test scale)
            step_ratio=0.5,  # Large step to force below minimum
            sample_size=3,
        )

        # Add metrics with long duration to trigger decrease
        for i in range(3):
            with suggester.record(batch_size=5):
                time.sleep(1.5)  # 1.5 seconds > 1.0 max_duration

        new_size = suggester.suggest_batch_size()
        assert new_size == 5  # Should not go below min_batch_size
        assert suggester.current_batch_size == 5

    def test_thread_safety(self):
        suggester = BatchSizeSuggester(sample_size=10)
        results = []

        def worker():
            for i in range(10):
                with suggester.record(batch_size=10):
                    time.sleep(0.001)
                results.append(suggester.suggest_batch_size())

        threads = [Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should not crash and should have recorded metrics
        assert len(suggester._history) > 0
        assert len(results) == 30  # 3 threads * 10 operations each

    def test_samples_preserves_history_after_batch_size_change(self):
        suggester = BatchSizeSuggester(
            sample_size=3,
            min_batch_size=5,  # Allow decrease
            min_duration=0.5,  # 0.5 seconds (test scale)
            max_duration=1.0,  # 1.0 seconds (test scale)
        )

        # Add initial fast metrics (under min_duration to trigger increase)
        for i in range(3):
            with suggester.record(batch_size=10):
                time.sleep(0.1)  # 0.1 seconds < 0.5 min_duration

        initial_history_count = len(suggester._history)

        # First suggestion should increase batch size due to fast execution
        old_batch_size = suggester.current_batch_size
        suggester.suggest_batch_size()

        # History should be preserved (not cleared)
        assert len(suggester._history) == initial_history_count

        # Verify batch size actually changed (increased)
        assert suggester.current_batch_size > old_batch_size

        # Add more metrics after batch size change
        for i in range(2):
            with suggester.record(batch_size=suggester.current_batch_size):
                time.sleep(0.1)

        # Samples should only include recent metrics (after batch size change)
        samples = suggester.samples
        assert len(samples) == 2  # Only metrics after batch size change
        assert all(m.executed_at >= suggester._batch_size_changed_at for m in samples)
