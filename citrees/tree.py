from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class _Node:
    """Node in decision tree.

    Parameters
    ----------
    index : int, optional (default=None)
        Index location of feature or column.

    p_value : float, optional (default=None)
        Probability value from feature selection.

    threshold : float, optional (default=None)
        Best split point found in feature.

    impurity : float, optional (default=None)
        Impurity measuring quality of split.

    estimate : Union[np.ndarray, float], optional (default=None)
        For classification trees, estimate of each class probability. For regression trees, central
        tendency estimate.

    left_child : Tuple[np.ndarray, np.ndarray], optional (default=None)
        First element a 2d array of features and second element a 1d array of labels.

    right_child : Tuple[np.ndarray, np.ndarray], optional (default=None)
        First element a 2d array of features and second element a 1d array of labels.
    """

    index: Optional[int] = None
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    impurity: Optional[float] = None
    estimate: Optional[Union[np.ndarray, float]] = None
    left_child: Optional[Tuple[np.ndarray, np.ndarray]] = None
    right_child: Optional[Tuple[np.ndarray, np.ndarray]] = None
