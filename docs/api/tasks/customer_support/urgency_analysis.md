# Urgency Analysis

Custom `urgency_levels` (level names), `response_times` (one time per level),
and `customer_tiers` (tier names) determine the allowed categorical values in
the returned schema. For example:

```python
from openaivec.task.customer_support import urgency_analysis

task = urgency_analysis(
    urgency_levels={"urgent": "Service outage", "routine": "General inquiry"},
    response_times={"urgent": "within_15_minutes", "routine": "within_2_days"},
    customer_tiers={"vip": "Priority account", "free": "Free plan"},
)
# task.response_format accepts "urgent", "within_15_minutes", and "vip".
```

All three choice sets must be nonempty, and `response_times` must have exactly
the same level keys as `urgency_levels`. With no custom choices, the original
`UrgencyAnalysis` model and defaults remain unchanged.

::: openaivec.task.customer_support.urgency_analysis
