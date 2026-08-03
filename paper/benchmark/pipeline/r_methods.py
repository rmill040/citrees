"""ctree/cforest wrappers through R via rpy2 for feature selection experiments.

This module provides Python wrappers for R's partykit package to enable
benchmarking against the original ctree (Hothorn et al., 2006) implementation.

The R_HOME environment variable is set dynamically based on the platform:
- macOS (homebrew): /opt/homebrew/Cellar/r/*/lib/R
- Docker/Linux: /usr/lib/R or /usr/lib64/R (Amazon Linux)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_R_MAX_SEED = int(np.iinfo(np.int32).max)
_MAX_SUPPORTED_RANDOM_STATE = 2 * _R_MAX_SEED


def _setup_r_home() -> None:
    """Set R_HOME environment variable if not already set."""
    if os.environ.get("R_HOME"):
        return

    # Linux paths (Debian/Ubuntu: /usr/lib/R, Amazon Linux: /usr/lib64/R)
    for r_path in ("/usr/lib/R", "/usr/lib64/R"):
        if Path(r_path).exists():
            os.environ["R_HOME"] = r_path
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


def _normalize_r_seed(random_state: int) -> int:
    """Map a nonnegative random state to a distinct seed accepted by R."""
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state must be an integer")

    value = int(random_state)
    if value < 0 or value > _MAX_SUPPORTED_RANDOM_STATE:
        raise ValueError(
            f"random_state must be between 0 and {_MAX_SUPPORTED_RANDOM_STATE}, got {value}"
        )
    if value <= _R_MAX_SEED:
        return value
    return value - (2 * _R_MAX_SEED + 1)


def _set_r_seed(ro: Any, random_state: int) -> None:
    """Configure reproducible serial and parallel R RNG streams."""
    ro.r["RNGkind"]("L'Ecuyer-CMRG", "Inversion", "Rejection")
    ro.r["set.seed"](_normalize_r_seed(random_state))


def _validate_inputs(X: np.ndarray, y: np.ndarray, task: str) -> None:
    """Validate the numeric matrix and target passed across the R boundary."""
    if task not in {"classification", "regression"}:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have different sample counts: {X.shape[0]} and {y.shape[0]}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must contain at least one sample and feature, got shape {X.shape}")
    try:
        finite_X = np.isfinite(X).all()
        finite_y = np.isfinite(y).all()
    except TypeError as exc:
        raise ValueError("X and y must contain numeric values") from exc
    if not finite_X or not finite_y:
        raise ValueError("X and y must contain only finite values")


def _resolve_mtry(mtry: int | str | None, n_features: int) -> int:
    """Resolve and validate partykit's number of candidate split features."""
    if mtry is None or mtry == "all":
        value = n_features
    elif mtry == "sqrt":
        value = int(np.ceil(np.sqrt(n_features)))
    elif mtry == "log":
        value = max(1, int(np.ceil(np.log2(n_features))))
    elif isinstance(mtry, int) and not isinstance(mtry, bool):
        value = mtry
    else:
        raise ValueError("mtry must be an integer, None, 'all', 'sqrt', or 'log'")

    if value < 1 or value > n_features:
        raise ValueError(f"mtry must be between 1 and {n_features}, got {value}")
    return value


def _resolve_cores(cores: int) -> int:
    """Resolve the R worker count, using all detected CPUs for -1."""
    if isinstance(cores, bool) or not isinstance(cores, int):
        raise TypeError("cores must be an integer")
    if cores == -1:
        return os.cpu_count() or 1
    if cores < 1:
        raise ValueError("cores must be -1 or a positive integer")
    return cores


def _ranking_from_named_scores(
    feature_names: list[str],
    scores: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Map named R feature scores to a complete zero-based ranking."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise RuntimeError(f"partykit returned a non-vector importance shape: {values.shape}")
    if len(feature_names) != len(values):
        raise RuntimeError(
            "partykit returned different numbers of variable names and importance values"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("partykit returned non-finite variable importance values")

    importance = np.zeros(n_features, dtype=np.float64)
    seen: set[int] = set()
    for name, value in zip(feature_names, values, strict=True):
        if not name.startswith("X") or not name[1:].isdigit():
            raise RuntimeError(f"Unexpected partykit feature name: {name!r}")
        feature_index = int(name[1:])
        if name != f"X{feature_index}":
            raise RuntimeError(f"Unexpected partykit feature name: {name!r}")
        if feature_index < 0 or feature_index >= n_features:
            raise RuntimeError(f"partykit feature index is out of range: {name!r}")
        if feature_index in seen:
            raise RuntimeError(f"partykit returned duplicate feature importance: {name!r}")
        importance[feature_index] = value
        seen.add(feature_index)

    return np.lexsort((np.arange(n_features), -importance))


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


def get_r_runtime_versions() -> dict[str, str]:
    """Return the R runtime and partykit versions used by the bridge."""
    ro, _importr = _import_rpy2()
    _get_partykit()
    return {
        "r": str(ro.r["R.version.string"][0]),
        "partykit": str(ro.r('as.character(packageVersion("partykit"))')[0]),
    }


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
    random_state: int = 1718,
) -> np.ndarray:
    """Fit ctree through rpy2 and return feature ranking based on variable usage.

    For a single tree, we rank features by how often they appear in splits.
    Features used more frequently are ranked higher; ties are broken by feature
    index.

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

    _validate_inputs(X, y, task)
    _set_r_seed(ro, random_state)

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

    # Extract variable usage via nodeapply + split_node + varid_split.
    # NOTE: tree[[id]]$split returns NULL because tree[[id]] yields a
    # constparty subtree, not the raw node.  The correct partykit API is
    # nodeapply(tree, FUN = function(n) varid_split(split_node(n))).
    r_code = """
    function(tree) {
        data_names <- names(data_party(tree))
        all_ids  <- nodeids(tree)
        term_ids <- nodeids(tree, terminal = TRUE)
        inner_ids <- setdiff(all_ids, term_ids)
        if (length(inner_ids) == 0) {
            return(setNames(numeric(0), character(0)))
        }
        varids <- nodeapply(tree, ids = inner_ids, FUN = function(n) {
            varid_split(split_node(n))
        })
        varids <- unlist(varids, use.names = FALSE)
        if (anyNA(varids) || any(varids < 1L) || any(varids > length(data_names))) {
            stop("partykit returned an invalid split-variable ID")
        }
        table(data_names[varids])
    }
    """
    get_var_counts = ro.r(r_code)
    var_counts = get_var_counts(tree)
    feature_names = [str(name) for name in ro.r["names"](var_counts)]
    return _ranking_from_named_scores(feature_names, np.asarray(var_counts), n_features)


def r_ctree_root_feature(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str = "Univariate",
    mincriterion: float = 0.0,
    minsplit: int = 2,
    minbucket: int = 1,
    random_state: int = 1718,
) -> int:
    """Fit a partykit ctree stump and return its zero-based root feature.

    A return value of -1 indicates that the fitted tree has no split.
    """
    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()
    n_features = X.shape[1]

    _validate_inputs(X, y, task)
    _set_r_seed(ro, random_state)
    r_data = _make_r_dataframe(X, y, task)
    control = partykit.ctree_control(
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        minsplit=minsplit,
        minbucket=minbucket,
        maxdepth=1,
    )
    tree = partykit.ctree(stats.as_formula("y ~ ."), data=r_data, control=control)

    r_code = """
    function(tree, n_features) {
        all_ids <- nodeids(tree)
        term_ids <- nodeids(tree, terminal = TRUE)
        inner_ids <- setdiff(all_ids, term_ids)
        if (length(inner_ids) == 0) return(-1L)
        vid <- unlist(nodeapply(tree, ids = inner_ids[1], FUN = function(n) {
            varid_split(split_node(n))
        }))[1]
        feature_names <- paste0("X", seq_len(n_features) - 1L)
        feature_index <- match(names(data_party(tree))[vid], feature_names)
        if (is.na(feature_index)) return(-1L)
        as.integer(feature_index - 1L)
    }
    """
    get_root_feature = ro.r(r_code)
    return int(np.asarray(get_root_feature(tree, n_features), dtype=np.int64)[0])


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
    random_state: int = 1718,
) -> np.ndarray:
    """Fit cforest through rpy2 and return feature ranking based on variable importance.

    Uses partykit's ``varimp()`` function for permutation-based variable
    importance.

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

    _validate_inputs(X, y, task)
    _set_r_seed(ro, random_state)

    mtry_val = _resolve_mtry(mtry, n_features)
    n_cores = _resolve_cores(cores)

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

    feature_names = [str(name) for name in ro.r["names"](varimp)]
    return _ranking_from_named_scores(feature_names, np.asarray(varimp), n_features)
