import argparse
import json
from pathlib import Path
from .eda import inspect_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()
    result = inspect_dataset(args.data_dir, args.output_dir)
    print(json.dumps(result, indent=2, default=str))
