# Data Folder

This folder is expected to contain the training and test images for the
autoencoder-based anomaly detection experiment.

The recommended structure mirrors the original project:

```text
data/
├─ train/
│  ├─ person1/
│  │   ├─ frame001.jpg
│  │   ├─ frame002.jpg
│  ├─ person2/
│  │   ├─ frame001.jpg
│  │   ├─ frame002.jpg
│  └─ ...
└─ test/
   ├─ person1/
   │   ├─ frame101.jpg
   │   ├─ frame102.jpg
   ├─ person2/
   │   ├─ frame101.jpg
   │   ├─ frame102.jpg
   └─ ...
```

- Images should be grayscale or color; they will be converted to grayscale and resized.
- Folder names (person1, person2, …) are used as labels for analysis (per-person anomaly rates).
