from typing import Tuple

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    """
    Handles normalization and flattening of images for the autoencoder.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def _flatten(self, X_imgs: np.ndarray) -> np.ndarray:
        """
        Flatten images from (n, H, W) to (n, H*W).
        """
        n, h, w = X_imgs.shape
        return X_imgs.reshape(n, h * w)

    def fit(self, X_imgs: np.ndarray) -> None:
        X_flat = self._flatten(X_imgs)
        self.scaler.fit(X_flat)

    def transform(self, X_imgs: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X_imgs)
        return self.scaler.transform(X_flat)

    def fit_transform(self, X_imgs: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X_imgs)
        return self.scaler.fit_transform(X_flat)

    def save(self, path: str) -> None:
        joblib.dump(self.scaler, path)

    def load(self, path: str) -> None:
        self.scaler = joblib.load(path)
