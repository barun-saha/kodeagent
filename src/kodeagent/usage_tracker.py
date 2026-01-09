"""Track and report LLM usage metrics (tokens and cost) across components."""

import asyncio

from .models import ComponentUsage, UsageMetrics


class UsageTracker:
    """Track cumulative LLM usage across components.

    Note: LLM usage is tracked only when calls are made via `call_llm` method. Any other LLM calls
    will not be tracked. E.g., if tools invoke LLMs directly, those calls will not be tracked.
    """

    def __init__(self):
        """Initialize the usage tracker."""
        self._usage_by_component: dict[str, ComponentUsage] = {}
        self._lock = asyncio.Lock()

    async def record_usage(self, component: str, metrics: UsageMetrics) -> None:
        """Record usage from a single LLM call. Uses lock to ensure coroutine safety.

        Args:
            component: Name of the component making the call.
            metrics: Usage metrics from the LLM call.
        """
        async with self._lock:
            if component not in self._usage_by_component:
                self._usage_by_component[component] = ComponentUsage(component_name=component)

            usage = self._usage_by_component[component]
            usage.call_count += 1
            usage.total_prompt_tokens += metrics.prompt_tokens
            usage.total_completion_tokens += metrics.completion_tokens
            usage.total_tokens += metrics.total_tokens
            usage.total_cost += metrics.cost

    def get_total_usage(self) -> ComponentUsage:
        """Get aggregated usage across all components.

        Returns:
            ComponentUsage with totals across all components.
        """
        total = ComponentUsage(component_name='Total')
        for usage in self._usage_by_component.values():
            total.call_count += usage.call_count
            total.total_prompt_tokens += usage.total_prompt_tokens
            total.total_completion_tokens += usage.total_completion_tokens
            total.total_tokens += usage.total_tokens
            total.total_cost += usage.total_cost
        return total

    def get_usage_by_component(self) -> dict[str, ComponentUsage]:
        """Get usage breakdown by component.

        Returns:
            Dictionary mapping component names to their usage.
        """
        return dict(self._usage_by_component)

    def format_report(self, include_breakdown: bool = True) -> str:
        """Generate a formatted usage report.

        Args:
            include_breakdown: Whether to include per-component breakdown.

        Returns:
            Formatted string with usage statistics.
        """
        total = self.get_total_usage()

        report_lines = [
            'LLM Usage Report',
            '=' * 60,
            f'Total LLM Calls: {total.call_count}',
            f'Total Tokens: {total.total_tokens:,}',
            f'  - Prompt Tokens: {total.total_prompt_tokens:,}',
            f'  - Completion Tokens: {total.total_completion_tokens:,}',
            f'Total Cost: ${total.total_cost:.4f}',
        ]

        if include_breakdown and self._usage_by_component:
            report_lines.append('')
            report_lines.append('Breakdown by Component:')
            report_lines.append('-' * 60)

            # Sort components by total cost (descending)
            sorted_components = sorted(
                self._usage_by_component.items(), key=lambda x: x[0], reverse=False
            )

            for component_name, usage in sorted_components:
                report_lines.append('')
                report_lines.append(f'{component_name}:')
                report_lines.append(
                    f'  Calls: {usage.call_count} | '
                    f'Tokens: {usage.total_tokens:,} | '
                    f'Cost: ${usage.total_cost:.4f}'
                )

        report_lines.append('=' * 60)
        return '\n'.join(report_lines)

    def reset(self) -> None:
        """Reset all usage tracking."""
        self._usage_by_component.clear()
