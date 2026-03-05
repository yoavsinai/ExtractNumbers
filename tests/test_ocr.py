import os
import cv2
import numpy as np
from scipy.io import loadmat

from src.ocr_engine import NumberExtractor


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw", "test_32x32.mat")

NUM_SAMPLES = 1000
RESIZE_DIM = 128


def load_svhn(path):
    data = loadmat(path)
    return data["X"], data["y"]


def get_image(X, y, index):
    img = X[:, :, :, index]
    label = y[index][0]
    if label == 10:
        label = 0
    return img, label


def evaluate():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    X, y = load_svhn(DATASET_PATH)
    extractor = NumberExtractor()

    correct = 0
    total = min(NUM_SAMPLES, X.shape[3])

    for i in range(total):
        img, true_label = get_image(X, y, i)

        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM),
                         interpolation=cv2.INTER_CUBIC)

        predicted = extractor.predict(img)

        if predicted is not None and predicted == true_label:
            correct += 1

        if i % 100 == 0:
            print(f"Processed {i}/{total}")

    print("\n======================")
    print("Accuracy:", correct / total)
    print("======================")


if __name__ == "__main__":
    evaluate()