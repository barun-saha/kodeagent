import os
import json
import argparse
from huggingface_hub import snapshot_download

REPO_ID = "gaia-benchmark/GAIA"
LOCAL_DIR = "gaia_dataset"

def download_gaia_dataset():
    """
    Downloads the GAIA dataset from Hugging Face Hub if not already present locally.
    """
    if os.path.exists(LOCAL_DIR):
        print(f"Using local saved copy of GAIA dataset from '{LOCAL_DIR}'.")
    else:
        print(f"Downloading GAIA dataset to '{LOCAL_DIR}'...")
        try:
            # The GAIA dataset is gated. You must be logged in to huggingface-cli
            # and have accepted the terms on the dataset's page.
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,  # Recommended for Windows
            )
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download dataset. You might need to log in to Hugging Face Hub.")
            print(f"Please run 'huggingface-cli login' and accept the terms for the dataset on its Hugging Face page.")
            print(f"Error: {e}")
            exit(1)

def iterate_questions(split: str):
    """
    Iterates through the GAIA dataset metadata, printing questions, answers, and associated files.

    Args:
        split (str): The dataset split to use, either 'test' or 'validation'.
    """
    if split not in ["test", "validation"]:
        raise ValueError("Split must be either 'test' or 'validation'")

    download_gaia_dataset()

    # The dataset files for the 2023 challenge are in a '2023' subdirectory.
    metadata_file = os.path.join(LOCAL_DIR, "2023", split, "metadata.jsonl")

    if not os.path.exists(metadata_file):
        print(f"Metadata file not found at {metadata_file}")
        return

    print(f"\n--- Processing {split} split ---")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            print("-" * 20)
            if 'Question' in data:
                print(f"Question: {data['Question']}")
            if 'Final answer' in data:
                print(f"Answer: {data['Final answer']}")
            if 'file_name' in data and data['file_name']:
                # The associated file is in the same directory as metadata.jsonl
                file_path = os.path.join(os.path.dirname(metadata_file), data['file_name'])
                print(f"File: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GAIA dataset.")
    parser.add_argument(
        "split",
        type=str,
        choices=["test", "validation"],
        help="The dataset split to process ('test' or 'validation').",
    )
    args = parser.parse_args()
    iterate_questions(args.split)
