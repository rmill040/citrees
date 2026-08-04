"""ctree/cforest wrappers through R via rpy2 for feature selection experiments.

This module provides Python wrappers for R's partykit package to enable
benchmarking against the original ctree (Hothorn et al., 2006) implementation.

The R_HOME environment variable is set dynamically based on the platform:
- macOS (homebrew): /opt/homebrew/Cellar/r/*/lib/R
- Docker/Linux: /usr/lib/R or /usr/lib64/R (Amazon Linux)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_R_MAX_SEED = int(np.iinfo(np.int32).max)
_MAX_SUPPORTED_RANDOM_STATE = 2 * _R_MAX_SEED
_R_MAX_PPSIZE = 500_000
_R_MAX_PPSIZE_OPTION = f"--max-ppsize={_R_MAX_PPSIZE}"


class RDiagnosticError(RuntimeError):
    """Raised when partykit returns malformed or unmappable diagnostics."""


@dataclass(frozen=True)
class RCTreeRootDiagnostics:
    """Feature-aligned diagnostics from a fitted partykit ctree root."""

    root_feature: int
    statistics: np.ndarray
    p_values: np.ndarray


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


def _configure_r_init_options(embedded: Any) -> None:
    """Require R's maximum supported protection stack before initialization."""
    options = tuple(embedded.get_initoptions())
    protection_stack_options = tuple(
        option for option in options if option.startswith("--max-ppsize")
    )
    if embedded.isinitialized():
        if protection_stack_options != (_R_MAX_PPSIZE_OPTION,):
            raise RuntimeError(
                "Embedded R was initialized without the required "
                f"{_R_MAX_PPSIZE_OPTION} option. Start a fresh process and import "
                "the citrees R adapter before rpy2.robjects."
            )
        return

    options_without_ppsize = tuple(
        option for option in options if not option.startswith("--max-ppsize")
    )
    embedded.set_initoptions((*options_without_ppsize, _R_MAX_PPSIZE_OPTION))


def _import_rpy2() -> tuple[Any, Any]:
    """Import rpy2 after configuring R_HOME and embedded-R initialization."""
    _setup_r_home()
    try:
        from rpy2.rinterface_lib import embedded  # type: ignore[import-not-found]

        _configure_r_init_options(embedded)
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
_root_diagnostics_function: Any | None = None

_R_CTREE_ROOT_DIAGNOSTICS = """
function(tree) {
    root_feature <- character(0)
    statistic_feature_names <- character(0)
    p_value_feature_names <- character(0)
    statistics <- numeric(0)
    p_values <- numeric(0)
    result <- function(error = character(0)) {
        list(
            root_feature = root_feature,
            statistic_feature_names = statistic_feature_names,
            p_value_feature_names = p_value_feature_names,
            statistics = statistics,
            p_values = p_values,
            error = error
        )
    }

    root_info <- info_node(node_party(tree))
    criterion <- root_info$criterion
    if (!is.null(criterion)) {
        required_rows <- c("statistic", "p.value")
        if (
            is.null(dim(criterion)) ||
            is.null(rownames(criterion)) ||
            is.null(colnames(criterion)) ||
            !all(required_rows %in% rownames(criterion))
        ) {
            return(result("partykit returned malformed root candidate diagnostics"))
        }
        statistic_values <- criterion["statistic", , drop = FALSE]
        p_value_values <- criterion["p.value", , drop = FALSE]
        statistic_feature_names <- colnames(statistic_values)
        p_value_feature_names <- colnames(p_value_values)
        statistics <- as.numeric(statistic_values)
        p_values <- as.numeric(p_value_values)
    }

    data_names <- names(data_party(tree))
    all_ids <- nodeids(tree)
    term_ids <- nodeids(tree, terminal = TRUE)
    inner_ids <- setdiff(all_ids, term_ids)
    if (length(inner_ids) > 0) {
        vid <- unlist(nodeapply(tree, ids = inner_ids[1], FUN = function(n) {
            varid_split(split_node(n))
        }), use.names = FALSE)
        if (
            length(vid) != 1L ||
            is.na(vid) ||
            vid < 1L ||
            vid > length(data_names)
        ) {
            return(result("partykit returned an invalid split-variable ID"))
        }
        root_feature <- data_names[[vid]]
    }

    result()
}
"""


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


def _feature_index_from_name(
    name: str,
    n_features: int,
    *,
    value_name: str,
) -> int:
    """Map one canonical R feature name to its zero-based Python index."""
    if not name.startswith("X") or not name[1:].isdigit():
        raise RDiagnosticError(f"Unexpected partykit {value_name} feature name: {name!r}")
    feature_index = int(name[1:])
    if name != f"X{feature_index}":
        raise RDiagnosticError(f"Unexpected partykit {value_name} feature name: {name!r}")
    if feature_index < 0 or feature_index >= n_features:
        raise RDiagnosticError(f"partykit {value_name} feature index is out of range: {name!r}")
    return feature_index


def _aligned_named_values(
    feature_names: list[str],
    values: np.ndarray,
    n_features: int,
    *,
    value_name: str,
    fill_value: float,
) -> np.ndarray:
    """Map named R values to their original zero-based feature positions."""
    numeric_values = np.asarray(values, dtype=np.float64)
    if numeric_values.ndim != 1:
        raise RDiagnosticError(
            f"partykit returned a non-vector {value_name} shape: {numeric_values.shape}"
        )
    if len(feature_names) != len(numeric_values):
        raise RDiagnosticError(
            f"partykit returned different numbers of {value_name} names and values"
        )
    if not np.isfinite(numeric_values).all():
        raise RDiagnosticError(f"partykit returned non-finite {value_name} values")

    aligned = np.full(n_features, fill_value, dtype=np.float64)
    seen: set[int] = set()
    for name, value in zip(feature_names, numeric_values, strict=True):
        feature_index = _feature_index_from_name(
            name,
            n_features,
            value_name=value_name,
        )
        if feature_index in seen:
            raise RDiagnosticError(f"partykit returned duplicate {value_name} feature: {name!r}")
        aligned[feature_index] = value
        seen.add(feature_index)

    return aligned


def _ranking_from_named_scores(
    feature_names: list[str],
    scores: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Map named R feature scores to a complete zero-based ranking."""
    importance = _aligned_named_values(
        feature_names,
        scores,
        n_features,
        value_name="importance",
        fill_value=0.0,
    )
    return np.lexsort((np.arange(n_features), -importance))


def _ranking_from_importance(importance: np.ndarray) -> np.ndarray:
    """Return a complete ranking with omitted importance values placed last."""
    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"importance must be one-dimensional, got shape {values.shape}")
    if np.isinf(values).any():
        raise ValueError("importance must not contain infinite values")
    ranking_values = np.where(np.isnan(values), -np.inf, values)
    return np.lexsort((np.arange(values.size), -ranking_values))


def _correct_monte_carlo_p_values(p_values: np.ndarray, n_resamples: int) -> np.ndarray:
    """Convert partykit's b/B values to the (b+1)/(B+1) convention."""
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    corrected = np.asarray(p_values, dtype=np.float64).copy()
    if corrected.ndim != 1:
        raise RDiagnosticError(
            f"partykit returned a non-vector candidate p-value shape: {corrected.shape}"
        )
    if np.isinf(corrected).any():
        raise RDiagnosticError("partykit returned non-finite candidate p-value values")

    finite = np.isfinite(corrected)
    finite_values = corrected[finite]
    if ((finite_values < 0.0) | (finite_values > 1.0)).any():
        raise RDiagnosticError("partykit returned a candidate p-value outside [0, 1]")

    extreme_counts = np.rint(finite_values * n_resamples)
    tolerance = np.finfo(np.float64).eps * max(1, n_resamples) * 8
    if not np.allclose(
        finite_values * n_resamples,
        extreme_counts,
        rtol=0.0,
        atol=tolerance,
    ):
        raise RDiagnosticError(
            f"partykit Monte Carlo p-values are not on the b/{n_resamples} lattice"
        )
    corrected[finite] = (extreme_counts + 1.0) / (n_resamples + 1.0)
    return corrected


def _root_diagnostics_from_named_values(
    *,
    root_feature_name: str | None,
    statistic_feature_names: list[str],
    p_value_feature_names: list[str],
    statistics: np.ndarray,
    p_values: np.ndarray,
    n_features: int,
) -> RCTreeRootDiagnostics:
    """Validate and align the candidate diagnostics returned by partykit."""
    if statistic_feature_names != p_value_feature_names:
        raise RDiagnosticError(
            "partykit returned unaligned candidate statistic and p-value feature names"
        )
    aligned_statistics = _aligned_named_values(
        statistic_feature_names,
        statistics,
        n_features,
        value_name="candidate statistic",
        fill_value=float("nan"),
    )
    aligned_p_values = _aligned_named_values(
        p_value_feature_names,
        p_values,
        n_features,
        value_name="candidate p-value",
        fill_value=float("nan"),
    )
    finite_p_values = aligned_p_values[np.isfinite(aligned_p_values)]
    if ((finite_p_values < 0.0) | (finite_p_values > 1.0)).any():
        raise RDiagnosticError("partykit returned a candidate p-value outside [0, 1]")

    root_feature = (
        -1
        if root_feature_name is None
        else _feature_index_from_name(
            root_feature_name,
            n_features,
            value_name="root",
        )
    )
    if root_feature >= 0 and (
        not np.isfinite(aligned_statistics[root_feature])
        or not np.isfinite(aligned_p_values[root_feature])
    ):
        raise RDiagnosticError("partykit root feature is missing candidate diagnostics")

    return RCTreeRootDiagnostics(
        root_feature=root_feature,
        statistics=aligned_statistics,
        p_values=aligned_p_values,
    )


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


def _get_root_diagnostics_function(ro: Any) -> Any:
    """Compile and cache the R closure used to inspect ctree roots."""
    global _root_diagnostics_function
    if _root_diagnostics_function is None:
        _root_diagnostics_function = ro.r(_R_CTREE_ROOT_DIAGNOSTICS)
    return _root_diagnostics_function


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


def r_ctree_root_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str = "MonteCarlo",
    nresample: int = 999,
    mincriterion: float = 0.0,
    minsplit: int = 2,
    minbucket: int = 1,
    random_state: int = 1718,
) -> RCTreeRootDiagnostics:
    """Fit a partykit ctree stump and return aligned root diagnostics.

    The defaults provide the fixed-budget Monte Carlo control used by the JSS
    cardinality experiment. Features omitted from partykit's root criterion,
    such as constant columns, retain their original positions with NaN values.
    Monte Carlo p-values use the corrected (b+1)/(B+1) convention.
    """
    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()

    _validate_inputs(X, y, task)
    n_features = X.shape[1]
    _set_r_seed(ro, random_state)
    r_data = _make_r_dataframe(X, y, task)
    control = partykit.ctree_control(
        teststat=teststat,
        testtype=testtype,
        nresample=nresample,
        mincriterion=mincriterion,
        minsplit=minsplit,
        minbucket=minbucket,
        maxdepth=1,
        saveinfo=True,
    )
    tree = partykit.ctree(stats.as_formula("y ~ ."), data=r_data, control=control)

    get_root_diagnostics = _get_root_diagnostics_function(ro)
    raw_diagnostics = get_root_diagnostics(tree)
    diagnostic_errors = [str(value) for value in raw_diagnostics.rx2("error")]
    if diagnostic_errors:
        raise RDiagnosticError("; ".join(diagnostic_errors))

    root_names = [str(name) for name in raw_diagnostics.rx2("root_feature")]
    if len(root_names) > 1:
        raise RDiagnosticError("partykit returned multiple root features")

    diagnostics = _root_diagnostics_from_named_values(
        root_feature_name=root_names[0] if root_names else None,
        statistic_feature_names=[
            str(name) for name in raw_diagnostics.rx2("statistic_feature_names")
        ],
        p_value_feature_names=[str(name) for name in raw_diagnostics.rx2("p_value_feature_names")],
        statistics=np.asarray(raw_diagnostics.rx2("statistics"), dtype=np.float64),
        p_values=np.asarray(raw_diagnostics.rx2("p_values"), dtype=np.float64),
        n_features=n_features,
    )
    if testtype != "MonteCarlo":
        return diagnostics
    return RCTreeRootDiagnostics(
        root_feature=diagnostics.root_feature,
        statistics=diagnostics.statistics,
        p_values=_correct_monte_carlo_p_values(diagnostics.p_values, nresample),
    )


def r_ctree_root_feature(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str = "Univariate",
    nresample: int = 999,
    mincriterion: float = 0.0,
    minsplit: int = 2,
    minbucket: int = 1,
    random_state: int = 1718,
) -> int:
    """Fit a partykit ctree stump and return its zero-based root feature.

    A return value of -1 indicates that the fitted tree has no split.
    """
    try:
        diagnostics = r_ctree_root_diagnostics(
            X,
            y,
            task=task,
            teststat=teststat,
            testtype=testtype,
            nresample=nresample,
            mincriterion=mincriterion,
            minsplit=minsplit,
            minbucket=minbucket,
            random_state=random_state,
        )
    except RDiagnosticError:
        return -1
    return diagnostics.root_feature


def r_cforest_importance(
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
    """Fit cforest through rpy2 and return aligned raw variable importance.

    Uses partykit's ``varimp()`` function for permutation-based variable
    importance. Named values preserve their original scale, sign, zeros, and
    ties. Features omitted by partykit are represented by NaN.

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
        Raw importance values aligned to the original Python feature order.
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
    return _aligned_named_values(
        feature_names,
        np.asarray(varimp),
        n_features,
        value_name="importance",
        fill_value=float("nan"),
    )


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
    """Fit cforest and rank features by aligned raw variable importance."""
    importance = r_cforest_importance(
        X,
        y,
        task=task,
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        ntree=ntree,
        mtry=mtry,
        replace=replace,
        fraction=fraction,
        varimp_conditional=varimp_conditional,
        varimp_nperm=varimp_nperm,
        cores=cores,
        random_state=random_state,
    )
    return _ranking_from_importance(importance)
