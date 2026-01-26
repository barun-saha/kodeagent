"""KodeAgent example demonstrating data cleaning and transformation of a CSV dataset."""

from pathlib import Path

from kodeagent import CodeActAgent, print_response
from kodeagent.tools import download_file

DATA_URL = 'https://data.cityofnewyork.us/resource/h9gi-nx95.csv?$limit=2000'

PERSONA = """
You are an expert data cleaning and transformation agent. You process one or more CSV files (or their URLs) provided to you.

Your job is to clean, transform, and generate a new CSV file as output based on the user instructions, ensuring data integrity and correctness.
You must also generate a data validation report in JSON format summarizing the cleaning operations performed and key statistics before and after cleaning.
This validation report is an audit trail of the cleaning process.

CRITICAL: You MUST handle all the CSV fields very carefully, ensuring NO data corruption or loss beyond what is specified in the user's task.
Careful when handling quotes, commas, and newlines in CSV fields.
Treat data types appropriately (e.g., strings, numbers, dates) based on the headers (when available) and content.
"""

TASK = """
You're given the URL of a CSV file. Your task is to:
- Add a new field to the data: crash_datetime = combine crash_date + crash_time, parse to ISO 8601 (YYYY-MM-DDTHH:MM:SS)
   - Make it the first column in the output CSV
- Drop these existing columns: 'location', 'crash_date', 'crash_time'
- Drop the rows where ANY of the following conditions are met:
   - 'latitude', 'longitude', and 'on_street_name' -- ALL these fields are missing/empty
   - 'latitude' out of [-90..90] range
   - 'longitude' out of [-180..180] range
   - 'crash_datetime' is missing/empty or cannot be parsed (in the transformed CSV)
- Normalize (after dropping rows and columns):
   - borough: strip whitespace; if empty/NaN, set to 'UNKNOWN'
   - latitude/longitude: set null or empty values to 0.0

Save the cleaned data to cleaned_data.csv. Review the cleaned file to verify that all transformations were applied correctly.

In addition, create a data validation report (validation_report.json) containing:
   - Original stats before cleaning
   - Final stats after cleaning
   - JSON keys:
       - input_rows, output_rows, duplicates_removed, rows_dropped_invalid_coords, rows_missing_datetime
       - null_counts for ['borough','latitude','longitude','crash_datetime'] after cleaning
       - row_indices_dropped: list of original row indices that were dropped and reason (as a dict)
       - schema: map column name → pandas dtype (string)

Also print a short summary at the end: what was done, how many rows removed, and why.

IMPORTANT: Save key resources and results/progress as intermediate checkpoints/artifacts where appropriate
to avoid unnecessary recomputations and I/O in case of re-runs. E.g., avoid re-downloading a file if already present.
Clean up any such checkpoints/artifacts at the end of the task.
"""


async def main():
    """Run the data cleaning and transformation task asynchronously."""
    # Ensure a local workspace exists (KodeAgent uses a temp dir if this is omitted or missing)
    # Use an absolute path for work_dir in real applications to avoid issues with relative paths
    work_dir = Path('demo_workspace')
    work_dir.mkdir(exist_ok=True)

    agent = CodeActAgent(
        name='Data Cleaning Agent',
        model_name='gemini/gemini-2.5-flash-lite',  # Swap via LiteLLM (e.g., 'openai/gpt-4.1')
        tools=[download_file],
        run_env='host',  # Use 'e2b' for sandboxed execution if preferred
        allowed_imports=[
            'numpy',
            'matplotlib',
            'pandas',
            'csv',
            'json',
            're',
            'pathlib',
            'io',
            'datetime',
            'urllib',
            'requests',
            'tempfile',
            'os',
        ],
        pip_packages='pandas',
        work_dir=str(work_dir),
        persona=PERSONA,
    )

    async for response in agent.run(
        task=TASK,
        files=[DATA_URL],
        max_iterations=20,  # Increase if needed for complex tasks
    ):
        # Stream only the final message for brevity
        print_response(response, only_final=True)

    print('=' * 40)
    print(f'Agent result:\n{agent.task.result}')
    print('=' * 40)
    artifacts = '\n'.join(f'- {f}' for f in agent.artifacts)
    print(f'Agent artifacts:\n{artifacts}')
    print('=' * 40)
    print(f'Agent plan:\n{agent.current_plan}')
    print('=' * 40)
    print(f'LLM usage:\n{agent.get_usage_report()}')
    print('=' * 40)
    print(f'Agent steps: {agent.task.steps_taken}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
