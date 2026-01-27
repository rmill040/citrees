"""Pipeline infrastructure for citrees feature selection experiments.

This module provides the core domain logic for running experiments:
- types: MethodConfig, ExperimentConfig, Result dataclasses
- grid: ExperimentGrid for managing configurations
- methods: Method registry with metadata
- stage1: Feature selection logic
- stage2: Downstream evaluation logic

Ray workers import from this module directly (no CLI dependencies).
"""

from paper.scripts.pipeline.grid import ExperimentGrid
from paper.scripts.pipeline.methods import (
    CLF_METHODS,
    EMBEDDING_METHODS,
    METHOD_INFO,
    REG_METHODS,
    THREADED_METHODS,
    MethodInfo,
    expand_method_configs,
    get_all_method_info,
    get_method_info,
    get_methods,
)
from paper.scripts.pipeline.types import (
    ExperimentConfig,
    MethodConfig,
    Result,
    StageType,
    StatusType,
    TaskType,
)

__all__ = [
    # Types
    "MethodConfig",
    "ExperimentConfig",
    "Result",
    "TaskType",
    "StageType",
    "StatusType",
    # Grid
    "ExperimentGrid",
    # Methods
    "MethodInfo",
    "METHOD_INFO",
    "CLF_METHODS",
    "REG_METHODS",
    "THREADED_METHODS",
    "EMBEDDING_METHODS",
    "get_methods",
    "expand_method_configs",
    "get_method_info",
    "get_all_method_info",
]
