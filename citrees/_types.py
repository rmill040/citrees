"""Centralized type definitions for citrees."""

from enum import StrEnum
from typing import Annotated, TypeAlias

from pydantic import Field

# Numeric Type Aliases
ProbabilityFloat: TypeAlias = Annotated[float, Field(gt=0.0, le=1.0)]
PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0)]
ConfidenceFloat: TypeAlias = Annotated[float, Field(gt=0.5, lt=1.0)]
HonestyFraction: TypeAlias = Annotated[float, Field(gt=0.0, lt=1.0)]
MinSamplesSplit: TypeAlias = Annotated[int, Field(ge=2)]


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
    BALANCED = "balanced"
    STRATIFIED = "stratified"


class EstimatorType(StrEnum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


# Compound Type Aliases
EarlyStoppingOption: TypeAlias = EarlyStopping | None
NResamplesOption: TypeAlias = NResamples | NonNegativeInt | None
MaxValuesOption: TypeAlias = MaxValuesMethod | float | int | None
BootstrapMethodOption: TypeAlias = BootstrapMethod | None
SamplingMethodOption: TypeAlias = SamplingMethod | None
