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


class ThresholdMethod(StrEnum):
    EXACT = "exact"
    RANDOM = "random"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"


class BootstrapMethod(StrEnum):
    BAYESIAN = "bayesian"
    CLASSIC = "classic"


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
type BootstrapMethodOption = BootstrapMethod | None
type SamplingMethodOption = SamplingMethod | None
