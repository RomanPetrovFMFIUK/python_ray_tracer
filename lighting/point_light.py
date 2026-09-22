from dataclasses import dataclass
from . import LightAmbient
import numpy.typing as npt
import numpy as np

@dataclass
class LightPoint(LightAmbient):
    position: npt.ArrayLike

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
