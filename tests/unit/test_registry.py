"""Tests for citrees._registry.py."""

import pytest

from citrees._registry import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    ClassifierSplitters,
    ClassifierSplitterTests,
    Registry,
    RegressorSelectors,
    RegressorSelectorTests,
    RegressorSplitters,
    RegressorSplitterTests,
    ThresholdMethods,
)


class TestRegistry:
    """Tests for the Registry class."""

    def test_registry_name(self):
        """Test registry name property."""
        reg = Registry("TestRegistry")
        assert reg.name == "TestRegistry"

    def test_register_and_retrieve(self):
        """Test registering and retrieving a callable."""
        reg = Registry("TestRegistry")

        @reg.register("my_func")
        def my_func(x):
            return x * 2

        assert "my_func" in reg.keys()
        assert reg["my_func"](5) == 10

    def test_duplicate_alias_raises(self):
        """Test that duplicate alias raises KeyError."""
        reg = Registry("TestRegistry")

        @reg.register("duplicate")
        def func1():
            pass

        with pytest.raises(KeyError, match="already exists"):

            @reg.register("duplicate")
            def func2():
                pass

    def test_missing_key_raises(self):
        """Test that missing key raises KeyError."""
        reg = Registry("TestRegistry")
        with pytest.raises(KeyError, match="not found"):
            _ = reg["nonexistent"]

    def test_keys_returns_list(self):
        """Test that keys() returns a list."""
        reg = Registry("TestRegistry")

        @reg.register("func1")
        def func1():
            pass

        @reg.register("func2")
        def func2():
            pass

        keys = reg.keys()
        assert isinstance(keys, list)
        assert "func1" in keys
        assert "func2" in keys


class TestPredefinedRegistries:
    """Tests for predefined registries."""

    def test_classifier_selectors_has_expected_keys(self):
        """Test ClassifierSelectors has expected keys."""
        keys = ClassifierSelectors.keys()
        assert "mc" in keys
        assert "mi" in keys
        assert "rdc" in keys

    def test_classifier_selector_tests_has_expected_keys(self):
        """Test ClassifierSelectorTests has expected keys."""
        keys = ClassifierSelectorTests.keys()
        assert "mc" in keys
        assert "mi" in keys
        assert "rdc" in keys

    def test_regressor_selectors_has_expected_keys(self):
        """Test RegressorSelectors has expected keys."""
        keys = RegressorSelectors.keys()
        assert "pc" in keys
        assert "dc" in keys
        assert "rdc" in keys

    def test_regressor_selector_tests_has_expected_keys(self):
        """Test RegressorSelectorTests has expected keys."""
        keys = RegressorSelectorTests.keys()
        assert "pc" in keys
        assert "dc" in keys
        assert "rdc" in keys

    def test_classifier_splitters_has_expected_keys(self):
        """Test ClassifierSplitters has expected keys."""
        keys = ClassifierSplitters.keys()
        assert "gini" in keys
        assert "entropy" in keys

    def test_classifier_splitter_tests_has_expected_keys(self):
        """Test ClassifierSplitterTests has expected keys."""
        keys = ClassifierSplitterTests.keys()
        assert "gini" in keys
        assert "entropy" in keys

    def test_regressor_splitters_has_expected_keys(self):
        """Test RegressorSplitters has expected keys."""
        keys = RegressorSplitters.keys()
        assert "mse" in keys
        assert "mae" in keys

    def test_regressor_splitter_tests_has_expected_keys(self):
        """Test RegressorSplitterTests has expected keys."""
        keys = RegressorSplitterTests.keys()
        assert "mse" in keys
        assert "mae" in keys

    def test_threshold_methods_has_expected_keys(self):
        """Test ThresholdMethods has expected keys."""
        keys = ThresholdMethods.keys()
        assert "exact" in keys
        assert "random" in keys
        assert "percentile" in keys
        assert "histogram" in keys

    def test_registered_functions_are_callable(self):
        """Test that registered functions are callable."""
        # Check a few examples
        assert callable(ClassifierSelectors["mc"])
        assert callable(ClassifierSplitters["gini"])
        assert callable(RegressorSplitters["mse"])
        assert callable(ThresholdMethods["exact"])
