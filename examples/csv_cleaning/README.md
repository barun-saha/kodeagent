# CSV Cleaning with CodeAct

A real-life example of data cleaning using the CodeActAgent to synthesize, execute, validate, and refine Python code for transforming a messy CSV dataset.


## Problem Statement

You are given a messy CSV file containing real-world data with inconsistent formatting, missing values, and invalid entries. Your task is to clean and normalize the data according to a schema, then validate the output.

## Objective

Demonstrate **data cleaning via program synthesis and self-check** using the CodeActAgent:
- The agent writes and executes Python code to transform raw data
- It detects schema failures and inconsistencies via validation
- It revises and re-runs code until all validations pass
- This showcases real agent behavior: planning → execution → validation → refinement
- Use agent `persona`

## Inputs

**Dataset:** NYC Motor Vehicle Collisions (sample 2,000 rows)
- **Source:** NYC Open Data
- **URL:** https://data.cityofnewyork.us/resource/h9gi-nx95.csv?$limit=2000
- **Format:** CSV with mixed date/time formats, inconsistent casing, missing coordinates, and duplicate records

## Expected Outputs

Three artifacts saved to the work directory:

1. A cleaned CSV file
   - Cleaned and normalized data adhering to the target schema
   - UTF-8 encoding with header row

2. Data validation report (JSON)
   - Input/output row counts and rows removed (by reason)
   - Duplicates removed count
   - Schema: column name → pandas dtype

3. The source code
   - The exact Python code the agent generated and executed
   - Serves as provenance and reproducibility documentation


## Getting Started

### Prerequisites

- Python 3.10+
- Install KodeAgent: `pip install kodeagent`
- Set your LLM API key (e.g., `GOOGLE_API_KEY` for Gemini via LiteLLM)

### Running the Demo

```shell
python data_transformation.py
```
This will execute the CodeActAgent to clean the CSV file, validate the output, and save the results to the specified work directory.