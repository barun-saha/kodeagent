"""CSV Analysis Agent.
Wraps KodeAgent's ReActAgent with tools for analysing a pandas DataFrame.
The agent reasons about what to investigate, calls tools, observes results,
and produces a structured list of narrative-ready findings.
"""

import contextvars
import json
import logging
import os
import warnings
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from .. import kutils as ku
from ..kodeagent import ReActAgent

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', 'system')
CSV_ANALYST_PROMPT_FILE = os.path.join(PROMPT_DIR, 'csv_analyst.txt')

with open(CSV_ANALYST_PROMPT_FILE, encoding='utf-8') as f:
    CSV_ANALYST_SYSTEM_PROMPT = f.read()


_agent_df_storage: contextvars.ContextVar[pd.DataFrame | None] = contextvars.ContextVar(
    'agent_df_storage', default=None
)
"""A task-safe context variable to store the loaded DataFrame for the current agent session."""


def _get_df() -> pd.DataFrame | str:
    """Retrieve the loaded DataFrame, or return an error string.

    Returns:
        The loaded DataFrame or an error message if not found.
    """
    df = _agent_df_storage.get()
    if df is None:
        return 'Error: DataFrame not loaded. You must call init_df_for_analysis first.'
    return df


def init_df_for_analysis(csv_file_path: str) -> str:
    """Initialize the DataFrame for analysis. This tool must be called first.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        A string indicating success or failure.
    """
    try:
        df = pd.read_csv(csv_file_path)
        _agent_df_storage.set(df)
        df_shape = df.shape
        logger.info('Loaded DataFrame with shape %s from %s', df_shape, csv_file_path)
        return f'df read successfully! Rows: {df_shape[0]}, Columns: {df_shape[1]}'
    except Exception as e:
        logger.error('Error reading df: %s', e)
        return f'Error reading df! {e}'


def get_df_schema() -> str:
    """Get the schema of the dataset: column names, types, cardinality,
    sample values, and missing value counts. Always call this first.

    Returns:
        A JSON string containing the dataset schema and summary.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique()
        n_missing = int(df[col].isna().sum())
        sample = df[col].dropna().head(3).tolist()
        rows.append(
            {
                'column': col,
                'type': dtype,
                'unique_values': n_unique,
                'missing': n_missing,
                'sample': sample,
            }
        )
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': rows,
    }
    return json.dumps(summary, default=str)


def assess_column(column: str) -> str:
    """Assess whether a column is worth analysing for a data story.
    Returns a judgment and reasoning about the column's narrative value.
    Use this when unsure whether to investigate a column further.

    Args:
        column: The column name to assess.

    Returns:
        A text assessment with judgment and reasoning about the column.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    if column not in df.columns:
        return f'Column "{column}" not found. Available: {list(df.columns)}'

    series = df[column].dropna()
    n_unique = series.nunique()
    n_total = len(series)
    dtype = str(series.dtype)
    sample = series.head(5).tolist()

    # Build assessment context
    context = {
        'column': column,
        'dtype': dtype,
        'unique_ratio': round(n_unique / max(n_total, 1), 3),
        'unique_count': n_unique,
        'total_count': n_total,
        'sample_values': sample,
    }
    return (
        f'Column assessment for "{column}":\n'
        f'{json.dumps(context, default=str)}\n\n'
        f'Use your judgment: is this column likely an ID, coordinate, '
        f'or system field? Or does it contain meaningful variation worth exploring?'
    )


def get_summary_stats(columns: str) -> str:
    """Get summary statistics for one or more numeric columns.
    Returns mean, median, std, min, max, and percentiles.

    Args:
        columns: Comma-separated column names, e.g. "price,quantity,age"

    Returns:
        A JSON string with summary statistics for the requested columns.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    col_list = [c.strip() for c in columns.split(',')]
    results = {}
    for col in col_list:
        if col not in df.columns:
            results[col] = f'Column "{col}" not found'
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) == 0:
            results[col] = 'No numeric values found'
            continue
        results[col] = {
            'mean': round(float(series.mean()), 2),
            'median': round(float(series.median()), 2),
            'std': round(float(series.std()), 2),
            'min': round(float(series.min()), 2),
            'max': round(float(series.max()), 2),
            'p25': round(float(series.quantile(0.25)), 2),
            'p75': round(float(series.quantile(0.75)), 2),
            'missing': int(df[col].isna().sum()),
        }
    return json.dumps(results, default=str)


def get_value_counts(column: str, top_n: int = 10) -> str:
    """Get frequency distribution for a categorical column.
    Shows the most common values and their counts.

    Args:
        column: The column name to analyse.
        top_n: Number of top values to return (default 10).

    Returns:
        A JSON string with the frequency distribution.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    if column not in df.columns:
        return f'Column "{column}" not found'

    vc = df[column].value_counts().head(top_n)
    total = len(df)
    result = {
        'column': column,
        'total_rows': total,
        'unique_values': int(df[column].nunique()),
        'top_values': [
            {
                'value': str(k),
                'count': int(v),
                'percentage': round(v / total * 100, 1),
            }
            for k, v in vc.items()
        ],
    }
    return json.dumps(result, default=str)


def find_trends(numeric_column: str, time_column: str) -> str:
    """Analyse how a numeric column changes over time.
    Returns direction, magnitude of change, and whether the trend reversed.

    Args:
        numeric_column: The numeric column to analyse.
        time_column: The time/date column to use as x-axis.

    Returns:
        A JSON string or error message describing the detected trend.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    if numeric_column not in df.columns:
        return f'Column "{numeric_column}" not found'
    if time_column not in df.columns:
        return f'Column "{time_column}" not found'

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_copy = df.copy()
            df_copy[time_column] = pd.to_datetime(
                df_copy[time_column], format='mixed', errors='coerce'
            )
        sorted_df = df_copy.sort_values(time_column)
        values = pd.to_numeric(sorted_df[numeric_column], errors='coerce').dropna().values

        if len(values) < 4:
            return 'Not enough data points for trend analysis (need at least 4)'

        start_val = round(float(values[0]), 2)
        end_val = round(float(values[-1]), 2)
        pct_change = (end_val - start_val) / (abs(start_val) + 1e-9) * 100

        # Check for reversal
        mid = len(values) // 2
        first_half = values[mid] - values[0]
        second_half = values[-1] - values[mid]
        reversed_direction = bool(first_half * second_half < 0)

        result = {
            'column': numeric_column,
            'time_column': time_column,
            'start_value': start_val,
            'end_value': end_val,
            'pct_change': round(pct_change, 1),
            'direction': 'up' if pct_change > 0 else 'down',
            'reversed_midway': reversed_direction,
            'first_half_direction': 'up' if first_half > 0 else 'down',
            'second_half_direction': 'up' if second_half > 0 else 'down',
            'data_points': len(values),
        }
        return json.dumps(result, default=str)

    except Exception as e:
        return f'Error analysing trend: {str(e)}'


def find_anomalies(column: str) -> str:
    """Find unusually high or low values in a numeric column.
    Returns count of outliers and the most extreme values.

    Args:
        column: The numeric column to check for anomalies.

    Returns:
        A JSON string or error message detailing detected outliers.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    if column not in df.columns:
        return f'Column "{column}" not found'

    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(series) < 10:
        return 'Not enough data for anomaly detection (need at least 10 rows)'

    mean = series.mean()
    std = series.std()
    if std == 0:
        return f'Column "{column}" has zero variance — all values are identical'

    z_scores = ((series - mean) / std).abs()
    outliers = series[z_scores > 2.5]

    result = {
        'column': column,
        'total_values': len(series),
        'anomalous_count': int(len(outliers)),
        'anomalous_percentage': round(len(outliers) / len(series) * 100, 1),
        'mean': round(float(mean), 2),
        'max_value': round(float(series.max()), 2),
        'min_value': round(float(series.min()), 2),
        'most_extreme_values': sorted(outliers.tolist(), key=abs, reverse=True)[:5],
    }
    return json.dumps(result, default=str)


def compare_groups(numeric_column: str, category_column: str) -> str:
    """Compare average values of a numeric column across categories.
    Useful for finding which groups are highest, lowest, or most surprising.

    Args:
        numeric_column: The numeric column to compare.
        category_column: The categorical column to group by.

    Returns:
        A JSON string comparing statistical metrics across groups.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    if numeric_column not in df.columns:
        return f'Column "{numeric_column}" not found'
    if category_column not in df.columns:
        return f'Column "{category_column}" not found'

    try:
        grouped = (
            pd.to_numeric(df[numeric_column], errors='coerce')
            .groupby(df[category_column])
            .agg(['mean', 'count'])
            .round(2)
        )
        grouped = grouped.sort_values('mean', ascending=False)

        top = grouped.index[0]
        bottom = grouped.index[-1]
        pct_diff = (
            (grouped['mean'].iloc[0] - grouped['mean'].iloc[-1])
            / (abs(grouped['mean'].iloc[-1]) + 1e-9)
            * 100
        )

        result = {
            'numeric_column': numeric_column,
            'category_column': category_column,
            'top_category': str(top),
            'top_avg': round(float(grouped['mean'].iloc[0]), 2),
            'bottom_category': str(bottom),
            'bottom_avg': round(float(grouped['mean'].iloc[-1]), 2),
            'pct_difference': round(float(pct_diff), 1),
            'all_groups': {
                str(k): {
                    'avg': round(float(v['mean']), 2),
                    'count': int(v['count']),
                }
                for k, v in grouped.iterrows()
            },
        }
        return json.dumps(result, default=str)

    except Exception as e:
        return f'Error comparing groups: {str(e)}'


def find_correlations(columns: str) -> str:
    """Find correlations between numeric columns.
    Returns pairs with strong positive or negative relationships.

    Args:
        columns: Comma-separated column names to correlate,
                    e.g. "sales,profit,units"

    Returns:
        A JSON string listing strong statistical correlations.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    col_list = [c.strip() for c in columns.split(',')]
    valid_cols = [c for c in col_list if c in df.columns]

    if len(valid_cols) < 2:
        return f'Need at least 2 valid columns. Found: {valid_cols}'

    try:
        numeric_df = df[valid_cols].apply(pd.to_numeric, errors='coerce')
        corr = numeric_df.corr()

        strong_pairs = []
        for i, col_a in enumerate(valid_cols):
            for col_b in valid_cols[i + 1 :]:
                val = corr.loc[col_a, col_b]
                if pd.isna(val):
                    continue
                if abs(val) >= 0.7:
                    strong_pairs.append(
                        {
                            'col_a': col_a,
                            'col_b': col_b,
                            'correlation': round(float(val), 3),
                            'direction': 'positive' if val > 0 else 'negative',
                            'strength': 'perfect'
                            if abs(val) > 0.99
                            else 'very strong'
                            if abs(val) > 0.9
                            else 'strong',
                        }
                    )

        result = {
            'columns_analysed': valid_cols,
            'strong_correlations': sorted(
                strong_pairs,
                key=lambda x: abs(x['correlation']),
                reverse=True,
            ),
            'total_pairs_checked': len(valid_cols) * (len(valid_cols) - 1) // 2,
        }
        return json.dumps(result, default=str)

    except Exception as e:
        return f'Error finding correlations: {str(e)}'


def sample_rows(filter_column: str, filter_value: str, n: int = 5) -> str:
    """Get sample rows matching a condition. Useful for investigating
    specific anomalies or verifying a pattern.

    Args:
        filter_column: Column to filter on.
        filter_value: Value to match (string comparison).
        n: Number of rows to return (default 5, max 20).

    Returns:
        A JSON string containing the sample rows.
    """
    df = _get_df()
    if isinstance(df, str):
        return df
    if filter_column not in df.columns:
        return f'Column "{filter_column}" not found'

    n = min(n, 20)
    try:
        mask = df[filter_column].astype(str).str.contains(str(filter_value), case=False, na=False, regex=False)
        subset = df[mask].head(n)
        if len(subset) == 0:
            return f'No rows found where {filter_column} contains "{filter_value}"'
        return subset.to_json(orient='records', default_handler=str)
    except Exception as e:
        return f'Error sampling rows: {str(e)}'


class CSVAnalysisAgent(ReActAgent):
    """An agent specializing in discovering patterns and insights from CSV data.

    Examples:
        Using a local file:
            .. code-block:: python

                agent = CSVAnalysisAgent()
                # Pass the file path (or URL) directly as a task file
                async for response in agent.run(task, files=['/path/to/data.csv']):
                    pass

        Using a URL:
            .. code-block:: python

                agent = CSVAnalysisAgent()
                async for response in agent.run(task, files=['https://example.com/data.csv']):
                    pass
    """

    def __init__(
        self,
        name: str = 'CSV Analyst',
        model_name: str = 'gemini/gemini-2.0-flash-lite',
        **kwargs: Any,
    ) -> None:
        """Initialize the CSVAnalysisAgent.

        Args:
            name: Name of the agent.
            model_name: The LLM model to use.
            **kwargs: Additional arguments passed to ReActAgent.
        """
        tools = [
            init_df_for_analysis,
            get_df_schema,
            assess_column,
            get_summary_stats,
            get_value_counts,
            find_trends,
            find_anomalies,
            compare_groups,
            find_correlations,
            sample_rows,
        ]

        # Merge tools if user provides more
        if 'tools' in kwargs:
            tools.extend(kwargs.pop('tools'))

        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            system_prompt=CSV_ANALYST_SYSTEM_PROMPT,
            **kwargs,
        )

    async def pre_run(self) -> None:
        """Pre-run hook to auto-load CSV files and yield initialization logs."""
        # Reset the per-agent DataFrame state
        _agent_df_storage.set(None)

        # Auto-load the first available CSV file if provided
        if self.task and self.task.files:
            for file_path in self.task.files:
                if file_path.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        _agent_df_storage.set(df)
                        logger.info('Auto-loaded CSV from %s', file_path)
                        yield self.response(
                            rtype='log',
                            value=(
                                f'Auto-loaded dataset from {file_path}. Rows: {df.shape[0]},'
                                f' Cols: {df.shape[1]}'
                            ),
                            channel='run',
                        )
                        break
                    except Exception as e:
                        logger.error('Failed to auto-load CSV %s: %s', file_path, e)
                        continue

        # Yield from super class pre_run
        async for response in super().pre_run():
            yield response


async def main() -> None:
    """Example usage of the CSVAnalysisAgent."""
    import argparse

    parser = argparse.ArgumentParser(description='Run the CSV Analysis Agent.')
    parser.add_argument('csv_path', help='Path or URL to the CSV file to analyze')
    args = parser.parse_args()

    agent = CSVAnalysisAgent(
        model_name='gemini/gemini-2.5-flash-lite',
        description='An agent that analyzes CSV files.',
        litellm_params={'api_key': os.getenv('GOOGLE_API_KEY')},
        max_iterations=15,
        # tracing_type='langsmith',
    )

    task = (
        'Analyse this CSV dataset and find the most interesting patterns for a data story.\n\n'
        'Your final answer MUST be a valid JSON object with this exact structure:\n\n'
        '{{\n'
        '  "dataset_summary": "one sentence describing the dataset",\n'
        '  "has_time_dimension": true or false,\n'
        '  "findings": [\n'
        '    {{\n'
        '      "type": "trend|anomaly|correlation|comparison|distribution",\n'
        '      "severity": 0.0-1.0,\n'
        '      "description": "plain language finding",\n'
        '      "columns_involved": ["col1"],\n'
        '      "data_slice": {{"key": "value"}}\n'
        '    }}\n'
        '  ],\n'
        '  "boring": false,\n'
        '  "boring_message": null\n'
        '}}\n\n'
        'Return ONLY the JSON object. No prose, no markdown, no explanation.'
    )
    task_files = [args.csv_path]

    async for response in agent.run(task, files=task_files):
        print(response)

    print(f'\n\n{agent.task.result}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())