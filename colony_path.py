import numpy as np

# read colony csv for specific event, load into class, and then generate the Stereotype from a set of 
# source EEG files (with annotations extracted, same as colony.py), collect epochs
# per for specific event and generate path, then export to sterotypes/{event} 
    # can load multiple subjects if want to, it's a CLI app
# WHEN GENERATING WEIGHTS, need to do (x - min(X)) / 99th percentile so that things scale correctly

class Sterotype:
    def __init__(self, weights: np.ndarray, step: int, windows: int):
        self.size = len(weights)
        self.weights = weights
        self.step = step
        self.windows = windows
        self._summation = np.zeros((self.size, self.windows))
        self._count = 0
    
    def fit(self, windowed_epoch: np.ndarray):
        if windowed_epoch.shape != (self.size, self.windows):
            raise ValueError(f"Expected epoch shape {(self.size, self.windows)}, got {windowed_epoch.shape}")
        self._summation += windowed_epoch * self.weights[:, np.newaxis]
        self._count += 1
    
    # jitter parameters...
        # delay?
        # max amplitude? how do we vary amplitude? scaled vs alternating low and high?
        # phase? how do we vary phase? scaled vs alternating low and high? constant?
        # wavelength? how do we vary wavelength? scaled vs alternating low and high?
        # random gaps in between? how do we vary lengths? constant vs gradient?
    def get_path(self):
        if self._count == 0:
            return None
        return self._summation / self._count