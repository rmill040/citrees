import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    metric: str
    models: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    friedman_stat: float
    friedman_pvalue: float
    ranks: dict[str, float]


class StatisticalAnalysis:
    def __init__(self, results_path: Path | str):
        with open(results_path) as f:
            self.results = json.load(f)

    def aggregate(self, metric: str = "accuracy") -> dict[str, dict[str, list[float]]]:
        agg: dict[str, dict[str, list[float]]] = {}

        for r in self.results:
            ds = r["dataset"]
            model = r["model"]
            value = r["metrics"].get(metric)

            if value is None:
                continue

            if ds not in agg:
                agg[ds] = {}
            if model not in agg[ds]:
                agg[ds][model] = []

            agg[ds][model].append(value)

        return agg

    def friedman_test(self, metric: str = "accuracy") -> ComparisonResult:
        agg = self.aggregate(metric)

        datasets = list(agg.keys())
        models = list(set(m for ds in agg.values() for m in ds.keys()))

        matrix = np.zeros((len(datasets), len(models)))
        for i, ds in enumerate(datasets):
            for j, model in enumerate(models):
                if model in agg[ds]:
                    matrix[i, j] = np.mean(agg[ds][model])
                else:
                    matrix[i, j] = np.nan

        valid_rows = ~np.any(np.isnan(matrix), axis=1)
        matrix = matrix[valid_rows]

        if len(matrix) < 3:
            raise ValueError("Need at least 3 datasets for Friedman test")

        stat, pvalue = stats.friedmanchisquare(*matrix.T)

        ranks_matrix = np.zeros_like(matrix)
        for i in range(len(matrix)):
            ranks_matrix[i] = stats.rankdata(-matrix[i])

        avg_ranks = ranks_matrix.mean(axis=0)
        ranks = {m: float(r) for m, r in zip(models, avg_ranks)}

        means = {
            m: float(np.nanmean([agg[ds].get(m, [np.nan]) for ds in datasets])) for m in models
        }
        stds = {
            m: float(np.nanstd([np.mean(agg[ds].get(m, [np.nan])) for ds in datasets]))
            for m in models
        }

        return ComparisonResult(
            metric=metric,
            models=models,
            means=means,
            stds=stds,
            friedman_stat=float(stat),
            friedman_pvalue=float(pvalue),
            ranks=ranks,
        )

    def summary_table(self, metric: str = "accuracy") -> str:
        agg = self.aggregate(metric)

        models = sorted(set(m for ds in agg.values() for m in ds.keys()))

        lines = [f"{'Dataset':<30} " + " ".join(f"{m:>12}" for m in models)]
        lines.append("-" * len(lines[0]))

        for ds in sorted(agg.keys()):
            row = f"{ds:<30} "
            for model in models:
                if model in agg[ds]:
                    mean = np.mean(agg[ds][model])
                    row += f"{mean:>12.4f} "
                else:
                    row += f"{'N/A':>12} "
            lines.append(row)

        return "\n".join(lines)
