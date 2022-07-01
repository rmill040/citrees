from citrees import _Node
import numpy as np


def test_non_terminal_node():
    """Test a non-terminal or leaf node."""
    kwargs = {
        "index": 0,
        "p_value": 0.5,
        "threshold": 2.5,
        "impurity": 1.0,
        "left_child": (
            np.ones(6).reshape(3, 2), np.zeros(6)
        ),
        "right_child": (
            np.ones(6).reshape(3, 2), np.zeros(6)
        )
    }

    import pdb; pdb.set_trace()
    node = _Node(**kwargs)

    for key, value in node.dict().items():
        if key not in kwargs:
            assert value is None
        else:
            assert value == kwargs[key]


def test_terminal_node():
    """Test a terminal or leaf node."""
    kwargs = {
        "estimate": 0.5,
    }

    node = _Node(**kwargs)

    for key, value in node.dict().items():
        if key not in kwargs:
            assert value is None
        else:
            assert value == kwargs[key]
