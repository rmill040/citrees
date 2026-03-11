"""Tests for citrees._types.py."""

import pytest
from pydantic import BaseModel, ValidationError

from citrees._types import (
    ConfidenceFloat,
    EarlyStopping,
    EstimatorType,
    HonestyFraction,
    MaxValuesMethod,
    MinSamplesSplit,
    NonNegativeFloat,
    NonNegativeInt,
    NResamples,
    PositiveInt,
    ProbabilityFloat,
    SamplingMethod,
    ThresholdMethod,
)

# =============================================================================
# STRENUM TESTS
# =============================================================================


class TestEarlyStopping:
    """Tests for EarlyStopping enum."""

    def test_values(self):
        """Test enum values."""
        assert EarlyStopping.ADAPTIVE == "adaptive"
        assert EarlyStopping.SIMPLE == "simple"

    def test_membership(self):
        """Test string membership."""
        assert "adaptive" in [e.value for e in EarlyStopping]
        assert "simple" in [e.value for e in EarlyStopping]

    def test_string_comparison(self):
        """Test string comparison works."""
        assert EarlyStopping.ADAPTIVE == "adaptive"
        assert EarlyStopping("adaptive") == EarlyStopping.ADAPTIVE


class TestNResamples:
    """Tests for NResamples enum."""

    def test_values(self):
        """Test enum values."""
        assert NResamples.MINIMUM == "minimum"
        assert NResamples.MAXIMUM == "maximum"
        assert NResamples.AUTO == "auto"

    def test_from_string(self):
        """Test construction from string."""
        assert NResamples("minimum") == NResamples.MINIMUM
        assert NResamples("maximum") == NResamples.MAXIMUM
        assert NResamples("auto") == NResamples.AUTO


class TestMaxValuesMethod:
    """Tests for MaxValuesMethod enum."""

    def test_values(self):
        """Test enum values."""
        assert MaxValuesMethod.SQRT == "sqrt"
        assert MaxValuesMethod.LOG2 == "log2"


class TestThresholdMethod:
    """Tests for ThresholdMethod enum."""

    def test_values(self):
        """Test enum values."""
        assert ThresholdMethod.EXACT == "exact"
        assert ThresholdMethod.RANDOM == "random"
        assert ThresholdMethod.PERCENTILE == "percentile"
        assert ThresholdMethod.HISTOGRAM == "histogram"

    def test_all_methods(self):
        """Test all methods are present."""
        methods = [e.value for e in ThresholdMethod]
        assert len(methods) == 4


class TestSamplingMethod:
    """Tests for SamplingMethod enum."""

    def test_values(self):
        """Test enum values."""
        assert SamplingMethod.STRATIFIED == "stratified"
        assert SamplingMethod.UNDERSAMPLE == "undersample"
        assert SamplingMethod.OVERSAMPLE == "oversample"


class TestEstimatorType:
    """Tests for EstimatorType enum."""

    def test_values(self):
        """Test enum values."""
        assert EstimatorType.CLASSIFIER == "classifier"
        assert EstimatorType.REGRESSOR == "regressor"


# =============================================================================
# PYDANTIC TYPE ALIAS TESTS
# =============================================================================


class TestProbabilityFloat:
    """Tests for ProbabilityFloat type alias."""

    def test_valid_values(self):
        """Test valid probability values."""

        class Model(BaseModel):
            value: ProbabilityFloat

        # Valid: (0, 1]
        Model(value=0.001)
        Model(value=0.5)
        Model(value=1.0)

    def test_invalid_zero(self):
        """Test that 0 is invalid (must be > 0)."""

        class Model(BaseModel):
            value: ProbabilityFloat

        with pytest.raises(ValidationError):
            Model(value=0.0)

    def test_invalid_greater_than_one(self):
        """Test that > 1 is invalid."""

        class Model(BaseModel):
            value: ProbabilityFloat

        with pytest.raises(ValidationError):
            Model(value=1.5)

    def test_invalid_negative(self):
        """Test that negative is invalid."""

        class Model(BaseModel):
            value: ProbabilityFloat

        with pytest.raises(ValidationError):
            Model(value=-0.1)


class TestPositiveInt:
    """Tests for PositiveInt type alias."""

    def test_valid_values(self):
        """Test valid positive integers."""

        class Model(BaseModel):
            value: PositiveInt

        Model(value=1)
        Model(value=100)

    def test_invalid_zero(self):
        """Test that 0 is invalid."""

        class Model(BaseModel):
            value: PositiveInt

        with pytest.raises(ValidationError):
            Model(value=0)

    def test_invalid_negative(self):
        """Test that negative is invalid."""

        class Model(BaseModel):
            value: PositiveInt

        with pytest.raises(ValidationError):
            Model(value=-1)


class TestNonNegativeInt:
    """Tests for NonNegativeInt type alias."""

    def test_valid_values(self):
        """Test valid non-negative integers."""

        class Model(BaseModel):
            value: NonNegativeInt

        Model(value=0)
        Model(value=1)
        Model(value=100)

    def test_invalid_negative(self):
        """Test that negative is invalid."""

        class Model(BaseModel):
            value: NonNegativeInt

        with pytest.raises(ValidationError):
            Model(value=-1)


class TestNonNegativeFloat:
    """Tests for NonNegativeFloat type alias."""

    def test_valid_values(self):
        """Test valid non-negative floats."""

        class Model(BaseModel):
            value: NonNegativeFloat

        Model(value=0.0)
        Model(value=0.5)
        Model(value=100.0)

    def test_invalid_negative(self):
        """Test that negative is invalid."""

        class Model(BaseModel):
            value: NonNegativeFloat

        with pytest.raises(ValidationError):
            Model(value=-0.1)


class TestConfidenceFloat:
    """Tests for ConfidenceFloat type alias."""

    def test_valid_values(self):
        """Test valid confidence values (0.5, 1.0)."""

        class Model(BaseModel):
            value: ConfidenceFloat

        Model(value=0.51)
        Model(value=0.95)
        Model(value=0.99)

    def test_invalid_at_boundary(self):
        """Test that boundaries are invalid."""

        class Model(BaseModel):
            value: ConfidenceFloat

        with pytest.raises(ValidationError):
            Model(value=0.5)  # Must be > 0.5

        with pytest.raises(ValidationError):
            Model(value=1.0)  # Must be < 1.0

    def test_invalid_below_half(self):
        """Test that < 0.5 is invalid."""

        class Model(BaseModel):
            value: ConfidenceFloat

        with pytest.raises(ValidationError):
            Model(value=0.3)


class TestHonestyFraction:
    """Tests for HonestyFraction type alias."""

    def test_valid_values(self):
        """Test valid honesty fraction values (0, 1)."""

        class Model(BaseModel):
            value: HonestyFraction

        Model(value=0.1)
        Model(value=0.5)
        Model(value=0.9)

    def test_invalid_at_boundary(self):
        """Test that boundaries are invalid."""

        class Model(BaseModel):
            value: HonestyFraction

        with pytest.raises(ValidationError):
            Model(value=0.0)  # Must be > 0

        with pytest.raises(ValidationError):
            Model(value=1.0)  # Must be < 1


class TestMinSamplesSplit:
    """Tests for MinSamplesSplit type alias."""

    def test_valid_values(self):
        """Test valid min_samples_split values (>= 2)."""

        class Model(BaseModel):
            value: MinSamplesSplit

        Model(value=2)
        Model(value=10)
        Model(value=100)

    def test_invalid_below_two(self):
        """Test that < 2 is invalid."""

        class Model(BaseModel):
            value: MinSamplesSplit

        with pytest.raises(ValidationError):
            Model(value=1)

        with pytest.raises(ValidationError):
            Model(value=0)
