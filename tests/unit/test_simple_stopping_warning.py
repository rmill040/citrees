"""Requesting simple stopping warns that its p-values are not calibrated."""

from __future__ import annotations

import warnings

import pytest

from citrees import ConditionalInferenceTreeClassifier


def test_simple_stopping_emits_warning():
    with pytest.warns(UserWarning, match="simple"):
        ConditionalInferenceTreeClassifier(early_stopping_selector="simple")


def test_adaptive_stopping_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ConditionalInferenceTreeClassifier(early_stopping_selector="adaptive")
