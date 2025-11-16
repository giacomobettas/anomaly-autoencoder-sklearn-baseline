import argparse

import cv2
import numpy as np

from src.preprocess import Preprocessor
from src.model import SklearnAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Infer anomaly score for a single image.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/autoencoder.pkl")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Fixed anomaly threshold on reconstruction error.")
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    img = cv2.resize(img, (64, 64)).astype("float32")
    img_batch = np.expand_dims(img, axis=0)  # (1, H, W)

    # Load scaler
    prep = Preprocessor()
    prep.load(args.scaler_path)
    X = prep.transform(img_batch)

    # Load model
    autoencoder = SklearnAutoencoder(input_dim=X.shape[1])
    autoencoder.load(args.model_path)

    # Compute error
    errors = autoencoder.reconstruction_error(X)
    mse = float(errors[0])
    is_anomaly = mse > args.threshold

    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Anomaly (threshold={args.threshold}): {is_anomaly}")


if __name__ == "__main__":
    main()
