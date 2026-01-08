"""Test the UsageTracker class."""

import pytest

from kodeagent.models import UsageMetrics
from kodeagent.usage_tracker import UsageTracker


@pytest.mark.asyncio
async def test_usage_tracker_format_report_with_breakdown():
    """Test that format_report includes breakdown when requested and data exists."""
    tracker = UsageTracker()
    metrics = UsageMetrics(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.01)
    await tracker.record_usage('ComponentA', metrics)

    report = tracker.format_report(include_breakdown=True)

    assert 'Breakdown by Component:' in report
    assert 'ComponentA:' in report
    assert 'Calls: 1' in report
    assert 'Cost: $0.0100' in report
    assert 'Total Cost: $0.0100' in report


@pytest.mark.asyncio
async def test_usage_tracker_format_report_no_breakdown():
    """Test that format_report excludes breakdown when requested."""
    tracker = UsageTracker()
    metrics = UsageMetrics(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.01)
    await tracker.record_usage('ComponentA', metrics)

    report = tracker.format_report(include_breakdown=False)

    assert 'Breakdown by Component:' not in report
    assert 'ComponentA:' not in report
    assert 'Total Cost: $0.0100' in report


@pytest.mark.asyncio
async def test_usage_tracker_concurrency():
    """Test thread safety of record_usage."""
    tracker = UsageTracker()
    metrics = UsageMetrics(total_tokens=1, cost=1.0)

    # Run 100 concurrent updates
    import asyncio

    tasks = [tracker.record_usage('Comp', metrics) for _ in range(100)]
    await asyncio.gather(*tasks)

    total = tracker.get_total_usage()
    assert total.call_count == 100
    assert total.total_cost == 100.0
