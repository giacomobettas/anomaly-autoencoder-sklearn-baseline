import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_images_by_person
from src.preprocess import Preprocessor
from src.model import SklearnAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate autoencoder and compute anomaly rates per person.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test data root (e.g. data/test)")
    parser.add_argument("--model_path", type=str, default="models/autoencoder.pkl")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl")
    parser.add_argument("--threshold_percentile", type=float, default=99.0,
                        help="Percentile of training-like errors used as anomaly threshold (approx.).")
    args = parser.parse_args()

    # Load test images
    X_test_imgs, y_test = load_images_by_person(args.test_dir)
    print(f"Loaded {X_test_imgs.shape[0]} test images from {args.test_dir}.")

    # Load scaler and transform
    prep = Preprocessor()
    prep.load(args.scaler_path)
    X_test = prep.transform(X_test_imgs)

    # Load model
    autoencoder = SklearnAutoencoder(input_dim=X_test.shape[1])
    autoencoder.load(args.model_path)

    # Compute reconstruction errors
    errors = autoencoder.reconstruction_error(X_test)

    # Threshold
    threshold = np.percentile(errors, args.threshold_percentile)
    print(f"Anomaly threshold (p{args.threshold_percentile}): {threshold:.6f}")

    # Global stats
    anomalies = errors > threshold
    print(f"Global anomaly rate: {anomalies.mean() * 100:.2f}%")

    # Per-person anomaly rates
    for person in np.unique(y_test):
        mask = y_test == person
        person_errors = errors[mask]
        person_anomalies = person_errors > threshold
        rate = person_anomalies.mean() * 100
        print(f"{person}: {rate:.2f}% anomalies (n={mask.sum()})")

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50, alpha=0.8)
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.4f}")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.title("Reconstruction error distribution (test)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
