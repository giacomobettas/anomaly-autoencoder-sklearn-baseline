import os
import cv2
import numpy as np

def generate_image(size=(64, 64), pattern="noise"):
    if pattern == "noise":
        return (np.random.rand(*size) * 255).astype("uint8")
    elif pattern == "circle":
        img = np.zeros(size, dtype="uint8")
        cv2.circle(img, (size[1]//2, size[0]//2), size[0]//3, 255, -1)
        return img
    elif pattern == "square":
        img = np.zeros(size, dtype="uint8")
        cv2.rectangle(img, (16, 16), (48, 48), 255, -1)
        return img
    else:
        raise ValueError("Unknown pattern")

def save_img(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def main():
    base = "data"

    persons = ["person1", "person2"]
    patterns = ["noise", "circle"]

    for split in ["train", "test"]:
        for p, pattern in zip(persons, patterns):
            for i in range(3):  # 3 tiny images per folder
                img = generate_image(pattern=pattern)
                save_img(f"{base}/{split}/{p}/frame{i:03d}.jpg", img)

    print("Synthetic dataset created under data/")

if __name__ == "__main__":
    main()
