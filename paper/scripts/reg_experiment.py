"""Regressor experiments."""
import os
from pathlib import Path

# import pandas as pd

# from citrees import ConditionalInferenceForestRegressor, ConditionalInferenceTreeRegressor


HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
DATASETS = [f for f in os.listdir(DATA_DIR) if f.startswith("reg_")]


def main() -> None:
    """Main script to run experiments for regressors."""
    for dataset in DATASETS:
        print(dataset)


if __name__ == "__main__":
    main()
