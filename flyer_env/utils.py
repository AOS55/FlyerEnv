import numpy as np
from typing import Tuple, List, Union, Sequence

# Useful types
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[
    np.ndarray,
    Tuple[Vector, Vector],
    Tuple[Matrix, Matrix],
    Tuple[float, float],
    List[Vector],
    List[Matrix],
    List[float],
]


def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
