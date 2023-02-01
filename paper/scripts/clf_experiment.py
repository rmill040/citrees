"""Classifier experiments."""
import os
from pathlib import Path

from loguru import logger
import pandas as pd

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier


HERE = Path(__file__).resolve()
DATA_DIR = HERE.parents[1] / "data"
DATASETS = [f for f in os.listdir(DATA_DIR) if f.startswith("clf_")]


def main() -> None:
    """Main script to run experiments for classifiers."""
    for dataset in DATASETS:
        X = pd.read_parquet(os.path.join(DATA_DIR, dataset))
        y = X.pop("y").astype(int)
        X = X.astype(float)
        print(dataset, X.shape)
        import pdb; pdb.set_trace()
        clf = ConditionalInferenceForestClassifier(n_jobs=-1, verbose=3, max_samples=0.5).fit(X, y)


if __name__ == "__main__":
    main()
