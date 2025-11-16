import os
from typing import Tuple, List

import cv2
import numpy as np


def load_images_by_person(root_dir: str,
                          image_size: Tuple[int, int] = (64, 64)
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from a root directory organized as:
        root_dir/
            person1/
                img1.jpg
                img2.jpg
                ...
            person2/
                img1.jpg
                ...

    Args:
        root_dir: path to the root folder (e.g. "data/train" or "data/test").
        image_size: (width, height) to resize images.

    Returns:
        X: array of shape (n_samples, H, W) with grayscale images.
        y: array of shape (n_samples,) with person labels (folder names).
    """
    images: List[np.ndarray] = []
    labels: List[str] = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    for person in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person)
        if not os.path.isdir(person_path):
            continue

        for fname in sorted(os.listdir(person_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, image_size)
            images.append(img.astype("float32"))
            labels.append(person)

    if len(images) == 0:
        raise RuntimeError(f"No images found in {root_dir}")

    X = np.stack(images, axis=0)  # (n_samples, H, W)
    y = np.array(labels)
    return X, y
