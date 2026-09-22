from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
from . import BaseObject

@dataclass
class Sphere(BaseObject):
    center: npt.ArrayLike
    radius: float
    color: npt.ArrayLike

    def __post_init__(self):
        self.center = np.array(self.center, dtype=np.float64)
        self.color = np.array(self.color, dtype=np.float64)
