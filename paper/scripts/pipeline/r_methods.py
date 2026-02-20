"""R ctree/cforest wrappers via rpy2 for feature selection experiments.

This module provides Python wrappers for R's partykit package to enable
benchmarking against the original ctree (Hothorn et al., 2006) implementation.

The R_HOME environment variable is set dynamically based on the platform:
- macOS (homebrew): /opt/homebrew/Cellar/r/*/lib/R
- Docker/Linux: /usr/lib/R
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _setup_r_home() -> None:
    """Set R_HOME environment variable if not already set."""
    if os.environ.get("R_HOME"):
        return

    # Docker/Linux path
    if Path("/usr/lib/R").exists():
        os.environ["R_HOME"] = "/usr/lib/R"
        return

    # macOS homebrew path - find the latest version
    homebrew_r = Path("/opt/homebrew/Cellar/r")
    if homebrew_r.exists():
        versions = sorted(homebrew_r.iterdir(), reverse=True)
        for v in versions:
            r_lib = v / "lib" / "R"
            if r_lib.exists():
                os.environ["R_HOME"] = str(r_lib)
                return

    # Intel Mac homebrew
    homebrew_r_intel = Path("/usr/local/Cellar/r")
    if homebrew_r_intel.exists():
        versions = sorted(homebrew_r_intel.iterdir(), reverse=True)
        for v in versions:
            r_lib = v / "lib" / "R"
            if r_lib.exists():
                os.environ["R_HOME"] = str(r_lib)
                return


def _import_rpy2() -> tuple[Any, Any]:
    """Import rpy2 after ensuring R_HOME is set."""
    _setup_r_home()
    try:
        import rpy2.robjects as ro  # type: ignore[import-not-found]
        from rpy2.robjects.packages import importr  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "rpy2 is required for the R baselines (r_ctree/r_cforest). "
            "Install the experiment dependencies and ensure R is installed."
        ) from e
    return ro, importr


# Import R packages (lazy load on first use)
_partykit: Any | None = None
_stats: Any | None = None


def _get_partykit() -> Any:
    global _partykit
    if _partykit is None:
        _ro, importr = _import_rpy2()
        _partykit = importr("partykit")
    return _partykit


def _get_stats() -> Any:
    global _stats
    if _stats is None:
        _ro, importr = _import_rpy2()
        _stats = importr("stats")
    return _stats


def _make_r_dataframe(X: np.ndarray, y: np.ndarray, task: str) -> Any:
    """Create an R data frame from numpy arrays."""
    ro, _importr = _import_rpy2()
    n_features = X.shape[1]

    # Create dict of R vectors
    data_dict = {}
    for i in range(n_features):
        data_dict[f"X{i}"] = ro.FloatVector(X[:, i])

    # Add y - use FactorVector for classification, FloatVector for regression
    if task == "classification":
        data_dict["y"] = ro.FactorVector(ro.IntVector(y.astype(np.int64)))
    else:
        data_dict["y"] = ro.FloatVector(y.astype(np.float64))

    return ro.DataFrame(data_dict)


def r_ctree_ranking(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str = "Bonferroni",
    alpha: float = 0.05,
    nresample: int = 9999,
    minsplit: int = 20,
    minbucket: int = 7,
    **kwargs: Any,
) -> np.ndarray:
    """Fit R ctree and return feature ranking based on variable usage.

    For a single tree, we rank features by how often they appear in splits.
    Features used more frequently (especially near the root) are ranked higher.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector.
    task : str
        Either "classification" or "regression".
    teststat : str
        Test statistic type: "quadratic" or "maximum".
    testtype : str
        P-value computation: "Bonferroni", "MonteCarlo", "Univariate", or "Teststatistic".
    alpha : float
        Significance level for splits.
    nresample : int
        Number of Monte Carlo permutations (only used when testtype="MonteCarlo").
    minsplit : int
        Minimum samples required to attempt a split.
    minbucket : int
        Minimum samples in terminal nodes.

    Returns
    -------
    np.ndarray
        Feature indices sorted by importance (descending).
    """
    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()
    n_features = X.shape[1]

    # Create R data frame
    r_data = _make_r_dataframe(X, y, task)

    # Build formula
    formula = stats.as_formula("y ~ .")

    # Build ctree_control
    control = partykit.ctree_control(
        teststat=teststat,
        testtype=testtype,
        alpha=alpha,
        nresample=nresample,
        minsplit=minsplit,
        minbucket=minbucket,
    )

    # Fit ctree
    tree = partykit.ctree(formula, data=r_data, control=control)

    # Extract variable usage from the tree using R code
    r_code = """
    function(tree, n_features) {
        var_counts <- rep(0, n_features)
        nodes <- nodeids(tree)
        for (id in nodes) {
            node <- tree[[id]]
            if (!is.null(node$split)) {
                varid <- node$split$varid
                if (!is.null(varid) && varid >= 1 && varid <= n_features) {
                    var_counts[varid] <- var_counts[varid] + 1
                }
            }
        }
        return(var_counts)
    }
    """
    get_var_counts = ro.r(r_code)
    var_counts = np.array(get_var_counts(tree, n_features))

    # Rank by usage count (descending), ties broken by index
    ranking = np.lexsort((np.arange(n_features), -var_counts))
    return ranking


def r_cforest_ranking(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str = "Univariate",
    mincriterion: float = 0.0,
    nresample: int = 9999,
    ntree: int = 100,
    mtry: int | str | None = None,
    replace: bool = False,
    fraction: float = 0.632,
    varimp_conditional: bool = False,
    varimp_nperm: int = 1,
    cores: int = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Fit R cforest and return feature ranking based on variable importance.

    Uses partykit's varimp() function which implements permutation-based
    variable importance (mean decrease in accuracy).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector.
    task : str
        Either "classification" or "regression".
    teststat : str
        Test statistic type: "quadratic" or "maximum".
    testtype : str
        P-value computation: "Bonferroni", "MonteCarlo", or "Univariate".
    mincriterion : float
        1 - alpha threshold for splits. 0 means no stopping (grow full trees).
    nresample : int
        Number of Monte Carlo permutations.
    ntree : int
        Number of trees in the forest.
    mtry : int | str | None
        Number of variables to sample at each split.
        "sqrt" for sqrt(n_features), "log" for log2(n_features),
        "all" or None for all features.
    replace : bool
        Whether to sample with replacement (bootstrap).
    fraction : float
        Fraction of samples to use when replace=False.
    varimp_conditional : bool
        Whether to compute conditional variable importance.
    varimp_nperm : int
        Number of permutations for variable importance.
    cores : int
        Number of CPU cores for parallel tree growing and varimp.
        -1 means use all available cores (via os.cpu_count()).

    Returns
    -------
    np.ndarray
        Feature indices sorted by importance (descending).
    """
    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()
    n_features = X.shape[1]

    # Handle mtry parameter
    if mtry is None or mtry == "all":
        mtry_val = n_features
    elif mtry == "sqrt":
        mtry_val = int(np.ceil(np.sqrt(n_features)))
    elif mtry == "log":
        mtry_val = int(np.ceil(np.log2(n_features)))
    elif isinstance(mtry, int):
        mtry_val = mtry
    else:
        mtry_val = int(np.ceil(np.sqrt(n_features)))

    # Create R data frame
    r_data = _make_r_dataframe(X, y, task)

    # Build formula
    formula = stats.as_formula("y ~ .")

    # Build ctree_control for cforest
    control = partykit.ctree_control(
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        saveinfo=False,
    )

    # Build perturb list
    perturb = ro.ListVector({"replace": replace, "fraction": fraction})

    # Resolve cores (-1 means all available)
    n_cores = os.cpu_count() if cores == -1 else cores

    # Fit cforest (parallel tree growing)
    forest = partykit.cforest(
        formula,
        data=r_data,
        control=control,
        ntree=ntree,
        mtry=mtry_val,
        perturb=perturb,
        cores=n_cores,
    )

    # Get variable importance (parallel per-tree computation)
    varimp = partykit.varimp(
        forest,
        conditional=varimp_conditional,
        nperm=varimp_nperm,
        cores=n_cores,
    )

    # Convert to numpy and rank (descending importance)
    importance = np.array(varimp)
    ranking = np.argsort(importance)[::-1]
    return ranking
