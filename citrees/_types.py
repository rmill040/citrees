"""Centralized type definitions for citrees."""

from enum import StrEnum
from typing import Annotated

from pydantic import Field

# Numeric Type Aliases
type ProbabilityFloat = Annotated[float, Field(gt=0.0, le=1.0)]
type PositiveInt = Annotated[int, Field(gt=0)]
type NonNegativeInt = Annotated[int, Field(ge=0)]
type NonNegativeFloat = Annotated[float, Field(ge=0.0)]
type ConfidenceFloat = Annotated[float, Field(gt=0.5, lt=1.0)]
type HonestyFraction = Annotated[float, Field(gt=0.0, lt=1.0)]
type MinSamplesSplit = Annotated[int, Field(ge=2)]


# StrEnums
class EarlyStopping(StrEnum):
    ADAPTIVE = "adaptive"
    SIMPLE = "simple"


class NResamples(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    AUTO = "auto"


class MaxValuesMethod(StrEnum):
    SQRT = "sqrt"
    LOG2 = "log2"


class ThresholdTest(StrEnum):
    """How the split test controls the error rate over the candidate thresholds.

    BONFERRONI runs one permutation test per candidate threshold at level
    alpha / K with a permutation budget scaled by K, so a node with K candidates
    costs on the order of K**2 / alpha statistic evaluations. MAXT runs a single
    permutation test on the minimum impurity over all K candidates (the max-type
    statistic of Westfall and Young, 1993), which controls the same familywise
    error rate at on the order of K / alpha evaluations.
    """

    BONFERRONI = "bonferroni"
    MAXT = "maxt"


class ThresholdMethod(StrEnum):
    EXACT = "exact"
    RANDOM = "random"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"


class SamplingMethod(StrEnum):
    STRATIFIED = "stratified"
    UNDERSAMPLE = "undersample"
    OVERSAMPLE = "oversample"


class EstimatorType(StrEnum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


# Compound Type Aliases
type EarlyStoppingOption = EarlyStopping | None
type NResamplesOption = NResamples | NonNegativeInt | None
type MaxValuesOption = MaxValuesMethod | float | int | None
type SamplingMethodOption = SamplingMethod | None
