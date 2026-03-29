# Task-Specific Agents

While KodeAgent provides general-purpose `ReActAgent`, `CodeActAgent`, and `FunctionCallingAgent`, it also supports creating specialized agents tailored for specific domains or tasks. These agents encapsulate specialized system prompts and tools to provide a more focused and robust experience for targeted use cases.

## CSVAnalysisAgent

The `CSVAnalysisAgent` is a specialized `ReActAgent` designed for deep exploration and analysis of structured data in CSV format. It combines a rigorous data analysis persona with a suite of specialized tools for pandas DataFrame manipulation.

### Key Features

*   **Auto-loading**: Automatically loads CSV files (local paths or URLs) provided in the task's `files` list.
*   **Task-Safe State**: Uses `contextvars` to manage the loaded DataFrame, ensuring thread-safety and session isolation.
*   **Specialized Persona**: Configured with an expert data analyst persona that focuses on statistical significance, patterns, and anomalies.
*   **Actionable Tools**: Includes tools for schema inspection, summary statistics, trend analysis, anomaly detection, categorical comparisons, and correlations.

### Quick Start

```python
from kodeagent.agents import CSVAnalysisAgent
import asyncio

async def main():
    agent = CSVAnalysisAgent(model_name='gemini/gemini-2.5-flash-lite')
    
    # You can provide a local CSV file path or a remote URL
    dataset_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'
    
    task = 'Carefully analyze this CSV file in detail. Report the key trends, anomalies, and insights.'
    
    async for response in agent.run(task, files=[dataset_url]):
        if response['type'] == 'final':
            print(f'\nResult: {response["value"]}')

if __name__ == '__main__':
    asyncio.run(main())
```

### Specialized Tools

The agent comes pre-configured with the following specialized tools:

*   `get_df_schema`: Provides column names, types, cardinality, and missing values.
*   `get_summary_stats`: Calculates mean, median, standard deviation, and quantiles.
*   `find_trends`: Analyzes numeric changes over a time-series column.
*   `find_anomalies`: Detects outliers using statistical Z-scores.
*   `compare_groups`: Compares averages across categorical segments.
*   `find_correlations`: Identifies statistical relationships between numeric pairs.
*   `assess_column`: Evaluates a column's narrative or analytical value.

---

### Task Description Examples

The `CSVAnalysisAgent` can be driven by tasks ranging from simple information requests to detailed, structured reporting requirements.

#### 1. Simple Exploratory Task
A general request for the agent to find all statistically significant patterns. See the above example for this type of prompt.

#### 2. Narrative/Storytelling Task
Directs the agent to craft a human-readable narrative based on findings.
> "Analyze this CSV dataset in detail. Find the key trends, anomalies, and any other insights/interesting patterns. Do not write a dry report but craft an engaging story, based ONLY on the facts found in the CSV file."

#### 3. Structured JSON Reporting
Useful for automated pipelines where the agent's output needs to be parsed by other systems.
> "Analyze this CSV dataset and find the most interesting patterns. Your final answer MUST be a valid JSON object with the following structure:
> ```json
> {{
>   "dataset_summary": "one sentence describing the dataset",
>   "has_time_dimension": true,
>   "findings": [
>     {{
>       "type": "trend|anomaly|correlation|comparison|distribution",
>       "severity": 0.8,
>       "description": "finding description",
>       "columns_involved": ["col1"],
>       "data_slice": {{"key": "value"}}
>     }}
>   ]
> }}
> ```
> Return ONLY the JSON object."