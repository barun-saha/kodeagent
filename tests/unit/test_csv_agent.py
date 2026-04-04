"""Unit tests for the CSVAnalysisAgent and its tools."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from kodeagent.agents.csv_agent import (
    CSVAnalysisAgent,
    _agent_df_storage,
    _get_df,
    assess_column,
    compare_groups,
    find_anomalies,
    find_correlations,
    find_trends,
    get_df_schema,
    get_summary_stats,
    get_value_counts,
    init_df_for_analysis,
    sample_rows,
)


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Provides a mocked dataset for testing the CSV tools.

    Returns:
        A pandas DataFrame with sample data for testing CSV analysis tools.
    """
    return pd.DataFrame(
        {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'date': [
                '2023-01-01',
                '2023-01-02',
                '2023-01-03',
                '2023-01-04',
                '2023-01-05',
                '2023-01-06',
                '2023-01-07',
                '2023-01-08',
                '2023-01-09',
                '2023-01-10',
                '2023-01-11',
                '2023-01-12',
            ],
            'sales': [
                100,
                150,
                120,
                200,
                300,
                130,
                140,
                160,
                180,
                200,
                1000,
                240,
            ],  # Contains one extreme outlier (1000)
            'category': ['A', 'A', 'B', 'B', 'A', 'C', 'A', 'B', 'C', 'A', 'C', 'B'],
        }
    )


@pytest.fixture
def loaded_df(mock_df) -> pd.DataFrame:
    """Fixture to auto-load the mock dataset inside the context.

    Args:
        mock_df: The mock DataFrame fixture.

    Yields:
        The loaded mock DataFrame.
    """
    token = _agent_df_storage.set(mock_df)
    yield mock_df
    _agent_df_storage.reset(token)


@pytest.fixture
def clear_storage() -> None:
    """Fixture to clear out the dataframe storage context.

    Yields:
        None, after clearing the storage context.
    """
    token = _agent_df_storage.set(None)
    yield
    _agent_df_storage.reset(token)


def test_get_df_error(clear_storage) -> None:
    """Test _get_df helper logic when dataset is absent.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    result = _get_df()
    assert isinstance(result, str)
    assert 'Error: DataFrame not loaded' in result


def test_get_df_success(loaded_df) -> None:
    """Test _get_df helper logic when dataset is present.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = _get_df()
    assert isinstance(result, pd.DataFrame)
    assert result.equals(loaded_df)


@patch('kodeagent.agents.csv_agent.pd.read_csv')
def test_init_df_for_analysis(mock_read_csv, mock_df, clear_storage) -> None:
    """Test manual loading of dataset into context storage.

    Args:
        mock_read_csv: Mock for pd.read_csv.
        mock_df: The mock DataFrame fixture.
        clear_storage: Fixture to clear DataFrame storage.
    """
    mock_read_csv.return_value = mock_df
    result = init_df_for_analysis('dummy.csv')
    assert 'successfully' in result
    assert _agent_df_storage.get().equals(mock_df)


@patch('kodeagent.agents.csv_agent.pd.read_csv')
def test_init_df_for_analysis_error(mock_read_csv, clear_storage) -> None:
    """Test correct error messaging on initialization failure.

    Args:
        mock_read_csv: Mock for pd.read_csv.
        clear_storage: Fixture to clear DataFrame storage.
    """
    mock_read_csv.side_effect = Exception('File missing')
    result = init_df_for_analysis('dummy.csv')
    assert 'Error reading df' in result


def test_get_df_schema(loaded_df) -> None:
    """Test schema extraction from dataset.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = get_df_schema()
    assert '"total_rows": 12' in result
    assert '"total_columns": 4' in result


def test_get_df_schema_uninitialized(clear_storage) -> None:
    """Test schema fetch against uninitialized dataset.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    result = get_df_schema()
    assert 'Error: DataFrame not loaded' in result


def test_assess_column(loaded_df) -> None:
    """Test column evaluation and recommendation string extraction.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = assess_column('sales')
    assert '"unique_count": 11' in result

    result_missing = assess_column('missing')
    assert 'not found' in result_missing


def test_assess_column_uninitialized(clear_storage) -> None:
    """Test tool gracefully handles missing backend DataFrame.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    result = assess_column('sales')
    assert 'Error: DataFrame not loaded' in result


def test_get_summary_stats(loaded_df) -> None:
    """Test numeric column statistical extraction mechanism.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = get_summary_stats('sales')
    assert '"max": 1000.0' in result

    result_missing = get_summary_stats('missing')
    assert 'not found' in result_missing

    result_non_numeric = get_summary_stats('category')
    assert 'No numeric values found' in result_non_numeric


def test_get_summary_stats_uninitialized(clear_storage) -> None:
    """Test missing DataFrame scenario.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    result = get_summary_stats('sales')
    assert 'Error: DataFrame not loaded' in result


def test_get_value_counts(loaded_df) -> None:
    """Test categorical distribution value counts.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = get_value_counts('category')
    assert '"value": "A"' in result

    # Missing column
    result_missing = get_value_counts('missing')
    assert 'not found' in result_missing


def test_get_value_counts_uninitialized(clear_storage) -> None:
    """Test value counts with uninitialized DataFrame.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    assert 'Error: DataFrame' in get_value_counts('category')


def test_find_trends(loaded_df) -> None:
    """Test discovery tool for evaluating data drifts and directions.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = find_trends('sales', 'date')
    assert '"start_value": 100.0' in result
    assert '"end_value": 240.0' in result

    result_missing = find_trends('missing', 'date')
    assert 'not found' in result_missing

    result_missing = find_trends('sales', 'missing')
    assert 'not found' in result_missing


def test_find_trends_edge_cases(clear_storage, mock_df) -> None:
    """Test find_trends edge cases.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
        mock_df: The mock DataFrame fixture.
    """
    assert 'Error: DataFrame' in find_trends('sales', 'date')

    _agent_df_storage.set(mock_df.head(3))
    assert 'Not enough data points' in find_trends('sales', 'date')

    _agent_df_storage.set(None)


@patch('kodeagent.agents.csv_agent.pd.to_datetime')
def test_find_trends_exception(mock_dt, loaded_df) -> None:
    """Test find_trends exception handling.

    Args:
        mock_dt: Mock for pd.to_datetime.
        loaded_df: Fixture providing a loaded DataFrame.
    """
    mock_dt.side_effect = Exception('Mock failure')
    assert 'Error analysing trend: Mock failure' in find_trends('sales', 'date')


def test_find_anomalies(loaded_df) -> None:
    """Test statistical anomaly detection algorithms on datasets.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = find_anomalies('sales')
    # Row 11 has sales 1000 which is > 2.5 std deviations from mean
    assert '"anomalous_count": 1' in result


def test_find_anomalies_edge_cases(clear_storage, mock_df) -> None:
    """Test find_anomalies edge cases.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
        mock_df: The mock DataFrame fixture.
    """
    assert 'Error' in find_anomalies('sales')

    _agent_df_storage.set(mock_df)
    assert 'not found' in find_anomalies('missing')

    _agent_df_storage.set(mock_df.head(5))
    assert 'Not enough data' in find_anomalies('sales')

    df_zero_var = mock_df.copy()
    df_zero_var['const'] = 50
    _agent_df_storage.set(df_zero_var)
    assert 'zero variance' in find_anomalies('const')
    _agent_df_storage.set(None)


def test_compare_groups(loaded_df) -> None:
    """Test comparison mapping tool between numerical and categorical bounds.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = compare_groups('sales', 'category')
    assert 'top_category' in result
    assert '"C": {"avg": 436.67' in result  # Because "C" has the 1000

    assert 'not found' in compare_groups('missing', 'category')
    assert 'not found' in compare_groups('sales', 'missing')


def test_compare_groups_edge_cases(clear_storage) -> None:
    """Test compare_groups edge cases.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    assert 'Error' in compare_groups('sales', 'category')


@patch('kodeagent.agents.csv_agent.pd.to_numeric')
def test_compare_groups_exception(mock_tonumeric, loaded_df) -> None:
    """Test compare_groups exception handling.

    Args:
        mock_tonumeric: Mock for pd.to_numeric.
        loaded_df: Fixture providing a loaded DataFrame.
    """
    mock_tonumeric.side_effect = Exception('Aggregation fail')
    assert 'Error comparing' in compare_groups('sales', 'category')


def test_find_correlations(loaded_df) -> None:
    """Test discovering pearson correlations between numerical columns.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = find_correlations('sales,id')
    assert 'strong_correlations' in result


def test_find_correlations_edge_cases(clear_storage) -> None:
    """Test find_correlations edge cases.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    assert 'Error' in find_correlations('sales,id')


def test_find_correlations_missing_cols(loaded_df) -> None:
    """Test find_correlations with missing columns.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    assert 'Need at least 2 valid columns' in find_correlations('sales,missing')


@patch('kodeagent.agents.csv_agent.pd.to_numeric')
def test_find_correlations_exception(mock_numeric, loaded_df) -> None:
    """Test find_correlations exception handling.

    Args:
        mock_numeric: Mock for pd.to_numeric.
        loaded_df: Fixture providing a loaded DataFrame.
    """
    mock_numeric.side_effect = Exception('Math error')
    assert 'Error finding correlations' in find_correlations('sales,id')


def test_sample_rows(loaded_df) -> None:
    """Test dynamic query execution over text categories.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = sample_rows('category', 'A')
    assert '"category":"A"' in result

    assert 'No rows found' in sample_rows('category', 'Z')


def test_sample_rows_missing(loaded_df) -> None:
    """Test sample checking behavior on nonexistent bounds.

    Args:
        loaded_df: Fixture providing a loaded DataFrame.
    """
    result = sample_rows('missing_col', 'A')
    assert 'not found' in result


def test_sample_rows_uninitialized(clear_storage) -> None:
    """Test sample_rows with uninitialized DataFrame.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    assert 'Error' in sample_rows('category', 'A')


@patch('kodeagent.agents.csv_agent.pd.Series.str.contains')
def test_sample_rows_exception(mock_contains, loaded_df) -> None:
    """Test sample_rows exception handling.

    Args:
        mock_contains: Mock for pd.Series.str.contains.
        loaded_df: Fixture providing a loaded DataFrame.
    """
    mock_contains.side_effect = Exception('String error')
    assert 'Error sampling' in sample_rows('category', 'A')


@pytest.mark.asyncio
async def test_agent_pre_run_autoload(mock_df, clear_storage) -> None:
    """Test CSVAnalysisAgent lifecycle correctly bootstrapping files into storage.

    Args:
        mock_df: The mock DataFrame fixture.
        clear_storage: Fixture to clear DataFrame storage.
    """
    agent = CSVAnalysisAgent(
        model_name='gemini/gemini-2.0-flash-lite', litellm_params={'api_key': 'dummy'}
    )
    # mock agent task
    agent.task = MagicMock()
    agent.task.files = ['test_data.csv']

    with patch('kodeagent.agents.csv_agent.pd.read_csv') as mock_read:
        mock_read.return_value = mock_df

        # Consuming the async generator
        async for _ in agent.pre_run():
            pass

        assert _agent_df_storage.get() is not None
        assert _agent_df_storage.get().equals(mock_df)


@pytest.mark.asyncio
async def test_agent_pre_run_no_csv(clear_storage) -> None:
    """Test CSVAnalysisAgent skips non-CSV properties when loading contexts.

    Args:
        clear_storage: Fixture to clear DataFrame storage.
    """
    agent = CSVAnalysisAgent(
        model_name='gemini/gemini-2.0-flash-lite',
        litellm_params={'api_key': 'dummy'},
        tools=[get_df_schema],  # Extends default tools list line 783
    )
    # mock agent task
    agent.task = MagicMock()
    agent.task.files = ['some_image.jpg']

    async for _ in agent.pre_run():
        pass

    assert _agent_df_storage.get() is None


@pytest.mark.asyncio
async def test_main_execution() -> None:
    """Test main function instantiates and executes cleanly."""
    import sys

    from kodeagent.agents.csv_agent import main

    with patch.object(sys, 'argv', ['csv_agent.py', 'test.csv']):

        async def mock_run_coro(self, *args, **kwargs):
            self.task = MagicMock()
            self.task.result = 'dummy_result'
            yield {'type': 'log', 'value': 'done'}

        with patch('kodeagent.agents.csv_agent.CSVAnalysisAgent.run', new=mock_run_coro):
            await main()