import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PerformanceMetric:
    duration: float


@dataclass
class BatchSizeSuggester:
    current_batch_size: int = 10
    min_batch_size: int = 10
    min_duration: float = 30.0
    max_duration: float = 60.0
    step_ratio: float = 0.1
    sample_size: int = 10
    _metrics: List[PerformanceMetric] = field(default_factory=list)

    @contextmanager
    def record(self):
        start_time = time.perf_counter()
        try:
            yield

        except:
            raise

        finally:
            duration = time.time() - start_time
            self._metrics.append(PerformanceMetric(duration))
            self._metrics = self._metrics[-1024:]

    @property
    def samples(self) -> List[PerformanceMetric]:
        return self._metrics[-self.sample_size :]

    @property
    def average_duration(self) -> float:
        if not self.samples:
            return 0.0
        return sum(metric.duration for metric in self.samples) / len(self.samples)

    def clear_metrics(self):
        self._metrics.clear()

    def suggest_batch_size(self) -> int:
        if len(self._metrics) < self.sample_size:
            return self.current_batch_size

        average_duration = self.average_duration

        if average_duration < self.min_duration:
            new_batch_size = int(self.current_batch_size * (1 + self.step_ratio))
            self.clear_metrics()

        elif average_duration > self.max_duration:
            new_batch_size = int(self.current_batch_size * (1 - self.step_ratio))
            self.clear_metrics()

        else:
            new_batch_size = self.current_batch_size

        new_batch_size = max(new_batch_size, self.min_batch_size)
        self.current_batch_size = new_batch_size

        return self.current_batch_size
