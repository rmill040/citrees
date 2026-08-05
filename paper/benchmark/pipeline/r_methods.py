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
_R_TEST_TYPES = frozenset({"Bonferroni", "MonteCarlo", "Univariate", "Teststatistic"})
_R_COMBINED_TEST_TYPES = frozenset({"Bonferroni", "MonteCarlo"})


class RDiagnosticError(RuntimeError):
    """Raised when partykit returns malformed or unmappable diagnostics."""


@dataclass(frozen=True)
class RCTreeRootDiagnostics:
    """Feature-aligned diagnostics from a fitted partykit ctree root."""

    root_feature: int
    statistics: np.ndarray
    p_values: np.ndarray


@dataclass(frozen=True)
class RCTreeBehavior:
    """Structure and predictions from one fitted partykit ctree."""

    root_feature: int
    split_counts: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray | None


@dataclass(frozen=True)
class RCForestBehavior:
    """Importance and predictions from one fitted partykit cforest."""

    importances: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray | None


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
        from rpy2.rinterface_lib import embedded

        _configure_r_init_options(embedded)
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
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
_split_usage_function: Any | None = None
_bootstrap_weights_function: Any | None = None

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

_R_CTREE_SPLIT_USAGE = """
function(tree) {
    root_feature <- character(0)
    feature_names <- character(0)
    split_counts <- numeric(0)
    result <- function(error = character(0)) {
        list(
            root_feature = root_feature,
            feature_names = feature_names,
            split_counts = split_counts,
            error = error
        )
    }

    data_names <- names(data_party(tree))
    all_ids <- nodeids(tree)
    term_ids <- nodeids(tree, terminal = TRUE)
    inner_ids <- setdiff(all_ids, term_ids)
    if (length(inner_ids) == 0) {
        return(result())
    }

    varids <- unlist(nodeapply(tree, ids = inner_ids, FUN = function(n) {
        varid_split(split_node(n))
    }), use.names = FALSE)
    if (
        anyNA(varids) ||
        any(varids < 1L) ||
        any(varids > length(data_names))
    ) {
        return(result("partykit returned an invalid split-variable ID"))
    }

    root_feature <- data_names[[varids[[1]]]]
    counts <- table(data_names[varids])
    feature_names <- names(counts)
    split_counts <- as.numeric(counts)
    result()
}
"""

_R_BOOTSTRAP_WEIGHTS = """
function(n_samples, n_trees) {
    vapply(
        seq_len(n_trees),
        function(i) {
            tabulate(
                sample.int(n_samples, size = n_samples, replace = TRUE),
                nbins = n_samples
            )
        },
        integer(n_samples)
    )
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
    if not np.isrealobj(X) or not np.isrealobj(y):
        raise ValueError("X and y must contain real values")
    try:
        finite_X = np.isfinite(X).all()
        finite_y = np.isfinite(y).all()
    except TypeError as exc:
        raise ValueError("X and y must contain numeric values") from exc
    if not finite_X or not finite_y:
        raise ValueError("X and y must contain only finite values")
    if task == "classification":
        int64 = np.iinfo(np.int64)
        integer_values = tuple(int(value) for value in y)
        if any(value < int64.min or value > int64.max for value in integer_values):
            raise ValueError("classification targets must fit in signed 64-bit integers")
        integer_y = np.asarray(integer_values, dtype=np.int64)
        if not np.array_equal(y, integer_y):
            raise ValueError("classification targets must contain integer class labels")
        if np.unique(integer_y).size < 2:
            raise ValueError("classification targets must contain at least two classes")


def _validate_prediction_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    task: str,
) -> None:
    """Validate paired train and prediction matrices passed to partykit."""
    _validate_inputs(X_train, y_train, task)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be two-dimensional, got shape {X_test.shape}")
    if X_test.shape[0] == 0:
        raise ValueError("X_test must contain at least one sample")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(
            "X_train and X_test have different feature counts: "
            f"{X_train.shape[1]} and {X_test.shape[1]}"
        )
    if not np.isrealobj(X_test):
        raise ValueError("X_test must contain real values")
    try:
        finite = np.isfinite(X_test).all()
    except TypeError as exc:
        raise ValueError("X_test must contain numeric values") from exc
    if not finite:
        raise ValueError("X_test must contain only finite values")


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
    """Rank raw importance descending after restoring omitted variables as zero."""
    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"importance must be one-dimensional, got shape {values.shape}")
    if np.isinf(values).any():
        raise ValueError("importance must not contain infinite values")
    ranking_values = np.where(np.isnan(values), 0.0, values)
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


def _get_split_usage_function(ro: Any) -> Any:
    """Compile and cache the R closure used to inspect ctree split usage."""
    global _split_usage_function
    if _split_usage_function is None:
        _split_usage_function = ro.r(_R_CTREE_SPLIT_USAGE)
    return _split_usage_function


def _get_bootstrap_weights_function(ro: Any) -> Any:
    """Compile and cache the R closure used for n-out-of-n bootstrap weights."""
    global _bootstrap_weights_function
    if _bootstrap_weights_function is None:
        _bootstrap_weights_function = ro.r(_R_BOOTSTRAP_WEIGHTS)
    return _bootstrap_weights_function


def _r_testtype(ro: Any, testtype: str | tuple[str, ...]) -> str | Any:
    """Convert one or more partykit test-distribution controls for rpy2."""
    if isinstance(testtype, str):
        values: tuple[str, ...] = (testtype,)
    elif isinstance(testtype, tuple) and all(isinstance(value, str) for value in testtype):
        values = testtype
    else:
        raise TypeError("testtype must be a string or tuple of strings")
    if not values or len(set(values)) != len(values):
        raise ValueError("testtype must contain one or more unique values")
    unsupported = set(values) - _R_TEST_TYPES
    if unsupported:
        raise ValueError(f"unsupported partykit testtype values: {sorted(unsupported)}")
    if len(values) > 1 and set(values) != _R_COMBINED_TEST_TYPES:
        raise ValueError("combined partykit testtype must contain Bonferroni and MonteCarlo")
    return values[0] if len(values) == 1 else ro.StrVector(values)


def get_r_runtime_versions() -> dict[str, str]:
    """Return the R runtime and numerical package versions used by the bridge."""
    ro, _importr = _import_rpy2()
    _get_partykit()
    return {
        "r": str(ro.r["R.version.string"][0]),
        "partykit": str(ro.r('as.character(packageVersion("partykit"))')[0]),
        "libcoin": str(ro.r('as.character(packageVersion("libcoin"))')[0]),
        "mvtnorm": str(ro.r('as.character(packageVersion("mvtnorm"))')[0]),
    }


def _make_r_dataframe(X: np.ndarray, y: np.ndarray, task: str) -> Any:
    """Create an R data frame from numpy arrays."""
    ro, _importr = _import_rpy2()
    data_dict = {f"X{i}": ro.FloatVector(X[:, i]) for i in range(X.shape[1])}

    if task == "classification":
        integer_y = y.astype(np.int64)
        classes = np.unique(integer_y)
        data_dict["y"] = ro.r["factor"](
            ro.StrVector([str(value) for value in integer_y]),
            levels=ro.StrVector([str(value) for value in classes]),
        )
    else:
        data_dict["y"] = ro.FloatVector(y.astype(np.float64))

    return ro.DataFrame(data_dict)


def _make_r_predictor_dataframe(X: np.ndarray, ro: Any) -> Any:
    """Create an R predictor-only data frame with canonical feature names."""
    return ro.DataFrame({f"X{i}": ro.FloatVector(X[:, i]) for i in range(X.shape[1])})


def _r_prediction_outputs(
    model: Any,
    X_test: np.ndarray,
    y_train: np.ndarray,
    task: str,
    *,
    ro: Any,
    stats: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return validated partykit predictions and aligned class probabilities."""
    new_data = _make_r_predictor_dataframe(X_test, ro)
    raw = stats.predict(model, newdata=new_data, type="response")
    if task == "regression":
        predictions = np.asarray(raw, dtype=np.float64).copy()
        if predictions.shape != (X_test.shape[0],) or not np.isfinite(predictions).all():
            raise RDiagnosticError("partykit returned invalid regression predictions")
        return predictions, None, None

    labels = [str(value) for value in ro.r["as.character"](raw)]
    try:
        predictions = np.asarray([int(value) for value in labels], dtype=np.int64)
    except ValueError as exc:
        raise RDiagnosticError("partykit returned non-integer classification labels") from exc
    expected_classes = np.unique(y_train.astype(np.int64))
    if (
        predictions.shape != (X_test.shape[0],)
        or not np.isin(
            predictions,
            expected_classes,
        ).all()
    ):
        raise RDiagnosticError("partykit returned invalid classification predictions")

    raw_probabilities = stats.predict(model, newdata=new_data, type="prob")
    probabilities = np.asarray(raw_probabilities, dtype=np.float64).copy()
    probability_labels = [str(value) for value in ro.r["colnames"](raw_probabilities)]
    try:
        probability_classes = np.asarray(
            [int(value) for value in probability_labels],
            dtype=np.int64,
        )
    except ValueError as exc:
        raise RDiagnosticError("partykit returned non-integer probability labels") from exc
    if (
        probabilities.shape != (X_test.shape[0], len(expected_classes))
        or not np.array_equal(probability_classes, expected_classes)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise RDiagnosticError("partykit returned invalid classification probabilities")
    return predictions, probabilities, probability_classes


def _ctree_split_usage(tree: Any, *, ro: Any, n_features: int) -> tuple[int, np.ndarray]:
    """Return the root and feature-aligned split counts from one ctree."""
    raw = _get_split_usage_function(ro)(tree)
    errors = [str(value) for value in raw.rx2("error")]
    if errors:
        raise RDiagnosticError("; ".join(errors))

    root_names = [str(name) for name in raw.rx2("root_feature")]
    if len(root_names) > 1:
        raise RDiagnosticError("partykit returned multiple root features")
    root_feature = (
        -1
        if not root_names
        else _feature_index_from_name(
            root_names[0],
            n_features,
            value_name="root",
        )
    )
    feature_names = [str(name) for name in raw.rx2("feature_names")]
    split_counts = _aligned_named_values(
        feature_names,
        np.asarray(raw.rx2("split_counts"), dtype=np.float64),
        n_features,
        value_name="split count",
        fill_value=0.0,
    )
    if (split_counts < 0).any() or not np.equal(split_counts, np.floor(split_counts)).all():
        raise RDiagnosticError("partykit returned invalid split counts")
    if (root_feature < 0) != (split_counts.sum() == 0):
        raise RDiagnosticError("partykit root and split counts are inconsistent")
    if root_feature >= 0 and split_counts[root_feature] < 1:
        raise RDiagnosticError("partykit root feature is absent from split counts")
    return root_feature, split_counts


def r_ctree_ranking(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "Bonferroni",
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
    testtype : str or tuple of str
        P-value computation controls. A tuple combines supported partykit controls.
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
        testtype=_r_testtype(ro, testtype),
        alpha=alpha,
        nresample=nresample,
        minsplit=minsplit,
        minbucket=minbucket,
    )

    # Fit ctree
    tree = partykit.ctree(formula, data=r_data, control=control)

    _root_feature, split_counts = _ctree_split_usage(
        tree,
        ro=ro,
        n_features=n_features,
    )
    return _ranking_from_importance(split_counts)


def _fit_r_ctree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "MonteCarlo",
    mincriterion: float = 0.95,
    nresample: int = 999,
    maxdepth: int | None = 3,
    minsplit: int = 20,
    minbucket: int = 7,
    random_state: int = 1718,
) -> tuple[Any, Any, Any, Any]:
    """Fit one ctree and return its R bridge objects."""
    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()
    _validate_inputs(X_train, y_train, task)
    _set_r_seed(ro, random_state)

    control_kwargs: dict[str, Any] = {
        "teststat": teststat,
        "testtype": _r_testtype(ro, testtype),
        "splittest": False,
        "mincriterion": mincriterion,
        "nresample": nresample,
        "minsplit": minsplit,
        "minbucket": minbucket,
        "saveinfo": False,
    }
    if maxdepth is not None:
        control_kwargs["maxdepth"] = maxdepth
    control = partykit.ctree_control(**control_kwargs)
    tree = partykit.ctree(
        stats.as_formula("y ~ ."),
        data=_make_r_dataframe(X_train, y_train, task),
        control=control,
    )
    return ro, partykit, stats, tree


def r_ctree_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "MonteCarlo",
    mincriterion: float = 0.95,
    nresample: int = 999,
    maxdepth: int | None = 3,
    minsplit: int = 20,
    minbucket: int = 7,
    random_state: int = 1718,
) -> None:
    """Fit one ctree without computing predictions or feature summaries."""
    _fit_r_ctree(
        X,
        y,
        task=task,
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
        random_state=random_state,
    )


def r_ctree_behavior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "MonteCarlo",
    mincriterion: float = 0.95,
    nresample: int = 999,
    maxdepth: int | None = 3,
    minsplit: int = 20,
    minbucket: int = 7,
    random_state: int = 1718,
) -> RCTreeBehavior:
    """Fit one ctree and return structure and held-out predictions."""
    _validate_prediction_inputs(X_train, y_train, X_test, task)
    ro, _partykit, stats, tree = _fit_r_ctree(
        X_train,
        y_train,
        task=task,
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
        random_state=random_state,
    )
    root_feature, split_counts = _ctree_split_usage(
        tree,
        ro=ro,
        n_features=X_train.shape[1],
    )
    predictions, probabilities, classes = _r_prediction_outputs(
        tree,
        X_test,
        y_train,
        task,
        ro=ro,
        stats=stats,
    )
    return RCTreeBehavior(
        root_feature=root_feature,
        split_counts=split_counts,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
    )


def r_ctree_root_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "MonteCarlo",
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
    if not isinstance(testtype, str):
        raise TypeError("root diagnostics require one partykit testtype string")
    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()

    _validate_inputs(X, y, task)
    n_features = X.shape[1]
    _set_r_seed(ro, random_state)
    r_data = _make_r_dataframe(X, y, task)
    control = partykit.ctree_control(
        teststat=teststat,
        testtype=_r_testtype(ro, testtype),
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


def _fit_r_cforest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    teststat: str,
    testtype: str | tuple[str, ...],
    mincriterion: float,
    nresample: int,
    ntree: int,
    mtry: int | str | None,
    maxdepth: int | None,
    minsplit: int,
    minbucket: int,
    replace: bool,
    fraction: float,
    cores: int,
    random_state: int,
) -> tuple[Any, Any, Any, Any, int]:
    """Fit one cforest and return its R bridge objects and resolved core count."""
    if not isinstance(replace, (bool, np.bool_)):
        raise TypeError("replace must be a boolean")
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(
        fraction,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("fraction must be a real number")
    fraction_value = float(fraction)
    if not np.isfinite(fraction_value) or not 0.0 < fraction_value <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    replace_value = bool(replace)

    ro, _importr = _import_rpy2()
    partykit = _get_partykit()
    stats = _get_stats()
    _validate_inputs(X, y, task)
    _set_r_seed(ro, random_state)

    mtry_value = _resolve_mtry(mtry, X.shape[1])
    n_cores = _resolve_cores(cores)
    control_kwargs: dict[str, Any] = {
        "teststat": teststat,
        "testtype": _r_testtype(ro, testtype),
        "splittest": False,
        "mincriterion": mincriterion,
        "nresample": nresample,
        "minsplit": minsplit,
        "minbucket": minbucket,
        "saveinfo": False,
    }
    if maxdepth is not None:
        control_kwargs["maxdepth"] = maxdepth
    control = partykit.ctree_control(**control_kwargs)
    forest_kwargs: dict[str, Any] = {
        "data": _make_r_dataframe(X, y, task),
        "control": control,
        "ntree": ntree,
        "mtry": mtry_value,
    }
    if n_cores > 1:
        forest_kwargs["cores"] = n_cores
    if replace_value and fraction_value == 1.0:
        forest_kwargs["weights"] = _get_bootstrap_weights_function(ro)(
            X.shape[0],
            ntree,
        )
    else:
        forest_kwargs["perturb"] = ro.ListVector(
            {"replace": replace_value, "fraction": fraction_value}
        )
    forest = partykit.cforest(stats.as_formula("y ~ ."), **forest_kwargs)
    return ro, partykit, stats, forest, n_cores


def r_cforest_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "MonteCarlo",
    mincriterion: float = 0.95,
    nresample: int = 999,
    ntree: int = 100,
    mtry: int | str | None = None,
    maxdepth: int | None = 3,
    minsplit: int = 20,
    minbucket: int = 7,
    replace: bool = True,
    fraction: float = 1.0,
    cores: int = 1,
    random_state: int = 1718,
) -> None:
    """Fit one cforest without computing predictions or variable importance."""
    _fit_r_cforest(
        X,
        y,
        task=task,
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        ntree=ntree,
        mtry=mtry,
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
        replace=replace,
        fraction=fraction,
        cores=cores,
        random_state=random_state,
    )


def _r_cforest_importance(
    forest: Any,
    *,
    ro: Any,
    partykit: Any,
    n_features: int,
    conditional: bool,
    nperm: int,
    cores: int,
) -> np.ndarray:
    """Return aligned raw variable importance from one fitted cforest."""
    kwargs: dict[str, Any] = {
        "conditional": conditional,
        "nperm": nperm,
    }
    if cores > 1:
        kwargs["cores"] = cores
    raw = partykit.varimp(forest, **kwargs)
    feature_names = [str(name) for name in ro.r["names"](raw)]
    return _aligned_named_values(
        feature_names,
        np.asarray(raw),
        n_features,
        value_name="importance",
        fill_value=float("nan"),
    )


def r_cforest_importance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "Univariate",
    mincriterion: float = 0.0,
    nresample: int = 9999,
    ntree: int = 100,
    mtry: int | str | None = None,
    maxdepth: int | None = None,
    minsplit: int = 20,
    minbucket: int = 7,
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
    testtype : str or tuple of str
        P-value computation controls. A tuple combines supported partykit controls.
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
    maxdepth : int or None
        Maximum tree depth. None uses partykit's default.
    minsplit : int
        Minimum samples required to attempt a split.
    minbucket : int
        Minimum samples in terminal nodes.
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
    ro, partykit, _stats, forest, n_cores = _fit_r_cforest(
        X,
        y,
        task=task,
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        ntree=ntree,
        mtry=mtry,
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
        replace=replace,
        fraction=fraction,
        cores=cores,
        random_state=random_state,
    )
    return _r_cforest_importance(
        forest,
        ro=ro,
        partykit=partykit,
        n_features=X.shape[1],
        conditional=varimp_conditional,
        nperm=varimp_nperm,
        cores=n_cores,
    )


def r_cforest_behavior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "MonteCarlo",
    mincriterion: float = 0.95,
    nresample: int = 999,
    ntree: int = 100,
    mtry: int | str | None = None,
    maxdepth: int | None = 3,
    minsplit: int = 20,
    minbucket: int = 7,
    replace: bool = True,
    fraction: float = 1.0,
    varimp_conditional: bool = False,
    varimp_nperm: int = 1,
    cores: int = 1,
    random_state: int = 1718,
) -> RCForestBehavior:
    """Fit one cforest and return importance and held-out predictions."""
    _validate_prediction_inputs(X_train, y_train, X_test, task)
    ro, partykit, stats, forest, n_cores = _fit_r_cforest(
        X_train,
        y_train,
        task=task,
        teststat=teststat,
        testtype=testtype,
        mincriterion=mincriterion,
        nresample=nresample,
        ntree=ntree,
        mtry=mtry,
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
        replace=replace,
        fraction=fraction,
        cores=cores,
        random_state=random_state,
    )
    importances = _r_cforest_importance(
        forest,
        ro=ro,
        partykit=partykit,
        n_features=X_train.shape[1],
        conditional=varimp_conditional,
        nperm=varimp_nperm,
        cores=n_cores,
    )
    predictions, probabilities, classes = _r_prediction_outputs(
        forest,
        X_test,
        y_train,
        task,
        ro=ro,
        stats=stats,
    )
    return RCForestBehavior(
        importances=importances,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
    )


def r_cforest_ranking(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    teststat: str = "quadratic",
    testtype: str | tuple[str, ...] = "Univariate",
    mincriterion: float = 0.0,
    nresample: int = 9999,
    ntree: int = 100,
    mtry: int | str | None = None,
    maxdepth: int | None = None,
    minsplit: int = 20,
    minbucket: int = 7,
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
        maxdepth=maxdepth,
        minsplit=minsplit,
        minbucket=minbucket,
        replace=replace,
        fraction=fraction,
        varimp_conditional=varimp_conditional,
        varimp_nperm=varimp_nperm,
        cores=cores,
        random_state=random_state,
    )
    return _ranking_from_importance(importance)
