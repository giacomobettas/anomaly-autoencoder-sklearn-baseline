import numpy as np

from src.model import SklearnAutoencoder


def test_autoencoder_smoke():
    n_samples = 10
    dim = 64  # pretend we flattened 8x8 images

    X = np.random.rand(n_samples, dim).astype("float32")

    ae = SklearnAutoencoder(input_dim=dim, hidden_layers=(32, 8, 32), max_iter=50)
    ae.fit(X)
    recon = ae.reconstruct(X)

    assert recon.shape == X.shape
    # reconstruction error is non-negative
    errors = ae.reconstruction_error(X)
    assert errors.shape == (n_samples,)
    assert np.all(errors >= 0.0)