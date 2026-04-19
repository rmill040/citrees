"""Run one mechanism-dimension sweep shard and save raw outputs.

This wraps the existing mechanism builders so local and EC2 runs can execute
one focused shard at a time without duplicating analysis logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper.scripts.analysis.build_mechanism_summary_tables import (
    FixedDesignSpec,
    build_ensemble_split_study,
    build_single_tree_split_study,
)


def _build_specs(task: str, n_features_values: list[int], n_informative_values: list[int]) -> list[FixedDesignSpec]:
    specs: list[FixedDesignSpec] = []
    if task == "classification":
        for n_informative in n_informative_values:
            for n_features in n_features_values:
                specs.append(
                    FixedDesignSpec(
                        name=f"make_classification_n250_p{n_features}_i{n_informative}",
                        kind="easy_shuffled_classification",
                        dataset_seed=4718 + n_features + 10000 * n_informative,
                        n_samples=250,
                        n_features=n_features,
                        n_informative=n_informative,
                        class_sep=2.0,
                        flip_y=0.0,
                        design_family="vary_dimension_fixed_signal",
                        task="classification",
                    )
                )
        return specs

    for n_informative in n_informative_values:
        for n_features in n_features_values:
            specs.append(
                FixedDesignSpec(
                    name=f"make_regression_n250_p{n_features}_i{n_informative}",
                    kind="easy_shuffled_regression",
                    dataset_seed=6718 + n_features + 10000 * n_informative,
                    n_samples=250,
                    n_features=n_features,
                    n_informative=n_informative,
                    noise=5.0,
                    design_family="vary_dimension_fixed_signal",
                    task="regression",
                )
            )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tree", "forest"), required=True)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--n-features", type=int, nargs="+", default=[100, 250, 500, 1000])
    parser.add_argument("--n-informative", type=int, nargs="+", default=[2])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, nargs="*", default=[100], help="Forest only")
    parser.add_argument("--n-jobs", type=int, default=1, help="Forest only")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--estimator-verbose", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(args.task, args.n_features, args.n_informative)

    if args.mode == "tree":
        runs, summary, counts = build_single_tree_split_study(
            specs,
            n_seeds=args.n_seeds,
            verbose=args.verbose,
            estimator_verbose=args.estimator_verbose,
        )
        outputs: dict[str, pd.DataFrame] = {
            f"{args.output_stem}_runs.csv": runs,
            f"{args.output_stem}_summary.csv": summary,
            f"{args.output_stem}_feature_counts.csv": counts,
        }
    else:
        frames_runs: list[pd.DataFrame] = []
        frames_summary: list[pd.DataFrame] = []
        frames_counts: list[pd.DataFrame] = []
        for n_estimators in args.n_estimators:
            runs, summary, counts = build_ensemble_split_study(
                specs,
                n_estimators=n_estimators,
                n_seeds=args.n_seeds,
                verbose=args.verbose,
                estimator_verbose=args.estimator_verbose,
                n_jobs=args.n_jobs,
            )
            for frame in (runs, summary, counts):
                frame["sweep_n_estimators"] = n_estimators
            frames_runs.append(runs)
            frames_summary.append(summary)
            frames_counts.append(counts)
        outputs = {
            f"{args.output_stem}_runs.csv": pd.concat(frames_runs, ignore_index=True),
            f"{args.output_stem}_summary.csv": pd.concat(frames_summary, ignore_index=True),
            f"{args.output_stem}_feature_counts.csv": pd.concat(frames_counts, ignore_index=True),
        }

    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
