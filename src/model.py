from typing import Sequence

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor


class SklearnAutoencoder:
    """
    Simple autoencoder implemented with scikit-learn's MLPRegressor.
    It learns to reconstruct the input; reconstruction error is used for anomaly detection.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] = (1024, 256, 64, 256, 1024),
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10,
        )

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X, X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Mean squared reconstruction error per sample.
        """
        recon = self.reconstruct(X)
        return np.mean((X - recon) ** 2, axis=1)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
