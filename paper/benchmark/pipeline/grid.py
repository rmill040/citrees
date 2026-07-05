"""ExperimentGrid for managing experiment configurations.

Provides a clean abstraction for iterating over experiment configurations
and filtering based on completion status.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from paper.benchmark.pipeline.methods import (
    get_full_method_configs,
    get_methods,
)
from paper.benchmark.pipeline.types import ExperimentConfig, MethodConfig, TaskType

# R methods crash with "protect(): protection stack overflow" on high-dimensional
# datasets. These (method_name, dataset) pairs are excluded from the grid.
_EXCLUDED: set[tuple[str, str]] = {
    ("r_ctree", "dexter"),
    ("r_cforest", "dexter"),
}

# Specific (method_label, dataset, seed) combos that consistently fail after
# multiple retries across different instance types (c6a, r5). These are excluded
# from the grid so the API queue reports 0 pending when everything else is done.
# These cells are intentionally excluded so known non-completions do not keep
# distributed queues open after the rest of the benchmark finishes.
_SKIPPED: set[tuple[str, str, int]] = {
    # r_ctree MonteCarlo (testtype="MonteCarlo") hangs/OOMs on high-dim datasets.
    # The Bonferroni config (9d1ca9c27dfc7f5e) completes fine for these same datasets.
    ("r_ctree__b6e09ceb0eb26367", "gisette", 3),
    ("r_ctree__b6e09ceb0eb26367", "isolet", 2),
    ("r_ctree__b6e09ceb0eb26367", "isolet", 3),
    # CIT-RDC on these specific splits never finished reliably in repeated runs.
    # We treat them as known skipped cells rather than open missing work.
    ("cit__2f00ba06d3fd6444", "gisette", 0),
    ("cit__2f00ba06d3fd6444", "gisette", 1),
    ("cit__2f00ba06d3fd6444", "gisette", 3),
    ("cit__2f00ba06d3fd6444", "orlraws10P", 1),
}


@dataclass
class ExperimentGrid:
    """Grid of experiment configurations for distributed execution.

    Generates all combinations of (method, dataset, seed) for a task.
    Supports filtering to exclude completed configurations.

    Parameters
    ----------
    task : TaskType
        Either "classification" or "regression".
    methods : list[MethodConfig]
        Method configurations to include.
    datasets : list[str]
        Dataset names to include.
    seeds : list[int]
        Seed indices to include.

    Examples
    --------
    >>> grid = ExperimentGrid.from_cli("classification")
    >>> len(grid)
    4200

    >>> for config in grid:
    ...     print(config)
    cit__muting/adult/seed0
    cit__muting/adult/seed1
    ...
    """

    task: TaskType
    methods: list[MethodConfig]
    datasets: list[str]
    seeds: list[int]

    def __iter__(self) -> Iterator[ExperimentConfig]:
        """Iterate over all experiment configurations."""
        for method in self.methods:
            for dataset in self.datasets:
                if (method.name, dataset) in _EXCLUDED:
                    continue
                for seed in self.seeds:
                    if (method.label, dataset, seed) in _SKIPPED:
                        continue
                    yield ExperimentConfig(
                        method=method,
                        dataset=dataset,
                        seed=seed,
                        task=self.task,
                    )

    def __len__(self) -> int:
        """Total number of configurations in the grid."""
        excluded = sum(1 for m in self.methods for d in self.datasets if (m.name, d) in _EXCLUDED)
        base = (len(self.methods) * len(self.datasets) - excluded) * len(self.seeds)
        skipped = sum(
            1
            for m in self.methods
            for d in self.datasets
            if (m.name, d) not in _EXCLUDED
            for s in self.seeds
            if (m.label, d, s) in _SKIPPED
        )
        return base - skipped

    def filter_pending(self, completed: set[tuple[str, str, int]]) -> list[ExperimentConfig]:
        """Get list of configurations not in the completed set.

        Parameters
        ----------
        completed : set[tuple[str, str, int]]
            Set of (method_label, dataset, seed) tuples already completed.

        Returns
        -------
        list[ExperimentConfig]
            List of pending configurations.
        """
        return list(self.iter_pending(completed))

    def iter_pending(self, completed: set[tuple[str, str, int]]) -> Iterator[ExperimentConfig]:
        """Iterate over configurations not in the completed set."""
        for cfg in self:
            if cfg.key not in completed:
                yield cfg

    def count_pending(self, completed: set[tuple[str, str, int]]) -> int:
        """Count configurations not in the completed set (no materialization)."""
        return sum(1 for cfg in self if cfg.key not in completed)

    def as_list(self) -> list[ExperimentConfig]:
        """Convert to a list of configurations."""
        return list(self)

    @classmethod
    def from_cli(
        cls,
        task: str,
        *,
        methods: str | None = None,
        datasets: str | None = None,
        seeds: str | None = None,
        source: Literal["all", "real", "synthetic"] = "all",
        n_seeds: int = 5,
        get_datasets_fn: Callable[..., list[str]] | None = None,
        max_configs_per_method: int | None = None,
    ) -> ExperimentGrid:
        """Build grid from CLI arguments.

        Parameters
        ----------
        task : str
            Task type: "classification" or "regression".
        methods : str, optional
            Comma-separated list of method names to filter.
        datasets : str, optional
            Comma-separated list of dataset names to filter.
        seeds : str, optional
            Comma-separated list of seed indices to filter.
        source : {"all", "real", "synthetic"}, default "all"
            Dataset source filter.
        n_seeds : int, default 5
            Total number of seeds (used when seeds not specified).
        get_datasets_fn : callable, optional
            Function to get dataset list. If None, uses adapters.data.get_datasets.
        max_configs_per_method : int, optional
            Limit the number of configs per method (useful for testing).
            If None, all configs are used.

        Returns
        -------
        ExperimentGrid
            Configured experiments.

        Raises
        ------
        ValueError
            If unknown methods, datasets, or invalid seeds are specified.
        """
        task: TaskType = task  # type: ignore[assignment]

        # Get method names for task
        method_names = get_methods(task)

        # Get full hyperparameter grid for all methods
        all_method_configs = get_full_method_configs(method_names, task)

        # Get all datasets for task
        if get_datasets_fn is None:
            from paper.benchmark.adapters.data import get_datasets as _get_datasets

            get_datasets_fn = _get_datasets

        all_datasets = get_datasets_fn(task, source=source)

        # Apply method filter
        if methods:
            method_filter = {m.strip() for m in methods.split(",") if m.strip()}
            available = {c.name for c in all_method_configs}
            unknown = method_filter - available
            if unknown:
                raise ValueError(
                    f"Unknown methods: {sorted(unknown)}. Available: {sorted(available)}"
                )
            all_method_configs = [c for c in all_method_configs if c.name in method_filter]

        # Apply max_configs_per_method limit (useful for testing)
        if max_configs_per_method is not None and max_configs_per_method > 0:
            from collections import defaultdict

            by_method: dict[str, list[MethodConfig]] = defaultdict(list)
            for cfg in all_method_configs:
                by_method[cfg.name].append(cfg)

            limited: list[MethodConfig] = []
            for _method_name, configs in by_method.items():
                limited.extend(configs[:max_configs_per_method])
            all_method_configs = limited

        # Apply dataset filter
        if datasets:
            dataset_filter = {d.strip() for d in datasets.split(",") if d.strip()}
            unknown = dataset_filter - set(all_datasets)
            if unknown:
                raise ValueError(
                    f"Unknown datasets: {sorted(unknown)}. "
                    f"Available (first 10): {all_datasets[:10]}"
                )
            all_datasets = [d for d in all_datasets if d in dataset_filter]

        # Parse seeds
        if seeds:
            seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
            bad = [s for s in seed_list if s < 0 or s >= n_seeds]
            if bad:
                raise ValueError(f"Seed indices out of range (0..{n_seeds - 1}): {bad}")
        else:
            seed_list = list(range(n_seeds))

        return cls(
            task=task,
            methods=all_method_configs,
            datasets=all_datasets,
            seeds=seed_list,
        )

    def summary(self) -> dict[str, int]:
        """Get summary statistics for the grid.

        Returns
        -------
        dict[str, int]
            Summary with keys: methods, datasets, seeds, total.
        """
        return {
            "methods": len(self.methods),
            "datasets": len(self.datasets),
            "seeds": len(self.seeds),
            "total": len(self),
        }
