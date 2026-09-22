from dataclasses import dataclass
from . import LightAmbient
import numpy.typing as npt
import numpy as np

@dataclass
class LightDirectional(LightAmbient):
    direction: npt.ArrayLike

    def __post_init__(self):
        self.direction = np.asarray(self.direction, dtype=np.float64)

