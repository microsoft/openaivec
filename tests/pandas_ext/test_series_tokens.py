"""pandas token counting tests."""

import pandas as pd

# ---------------------------------------------------------------------------
# count_tokens batch optimization
# ---------------------------------------------------------------------------


class TestCountTokensBatch:
    def test_count_tokens_returns_correct_counts(self):

        from openaivec import pandas_ext  # noqa: F401 — registers .ai accessor
        from openaivec._provider import ensure_default_registrations

        ensure_default_registrations()

        s = pd.Series(["hello", "hello world", ""])
        result = s.ai.count_tokens()
        assert len(result) == 3
        assert result.iloc[0] > 0
        assert result.iloc[1] > result.iloc[0]
        assert result.iloc[2] == 0

    def test_count_tokens_preserves_index(self):

        from openaivec import pandas_ext  # noqa: F401 — registers .ai accessor
        from openaivec._provider import ensure_default_registrations

        ensure_default_registrations()

        s = pd.Series(["a", "bb", "ccc"], index=[10, 20, 30])
        result = s.ai.count_tokens()
        assert list(result.index) == [10, 20, 30]
        assert result.name == "num_tokens"
