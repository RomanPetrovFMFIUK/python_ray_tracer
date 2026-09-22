import numpy as np

# Canvas Resolution
WIDTH: int = 800
HEIGHT: int = 400

# Viewport Settings
d = 0.5
Vw = WIDTH / HEIGHT
Vh = 1.0

CAMERA_POSITION = np.array([-5, 2.0, -5], dtype=np.float64)

BACKGROUND_COLOR = (15.0, 8.0, 45.0)

CAMERA_ANGEL_X = np.radians(0)
CAMERA_ANGEL_Y = np.radians(45)
CAMERA_ANGEL_Z = np.radians(0)
