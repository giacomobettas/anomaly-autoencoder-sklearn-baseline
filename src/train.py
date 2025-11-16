import argparse
import os

import numpy as np

from src.data_loader import load_images_by_person
from src.preprocess import Preprocessor
from src.model import SklearnAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Train scikit-learn autoencoder for anomaly detection.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data root (e.g. data/train)")
    parser.add_argument("--model_path", type=str, default="models/autoencoder.pkl")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Load training images
    X_train_imgs, y_train = load_images_by_person(args.train_dir)
    print(f"Loaded {X_train_imgs.shape[0]} training images from {args.train_dir}, "
          f"{len(np.unique(y_train))} persons.")

    # Preprocess
    prep = Preprocessor()
    X_train = prep.fit_transform(X_train_imgs)

    # Build and train autoencoder
    input_dim = X_train.shape[1]
    autoencoder = SklearnAutoencoder(input_dim=input_dim)
    autoencoder.fit(X_train)

    # Save model and scaler
    autoencoder.save(args.model_path)
    prep.save(args.scaler_path)
    print(f"Model saved to {args.model_path}")
    print(f"Scaler saved to {args.scaler_path}")


if __name__ == "__main__":
    main()
