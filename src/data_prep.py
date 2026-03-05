import os
import requests

DATASET_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
RAW_DATA_DIR = "data/raw"
SAVE_PATH = os.path.join(RAW_DATA_DIR, "test_32x32.mat")


def download_dataset():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if os.path.exists(SAVE_PATH):
        print("Dataset already exists.")
        return

    print("Downloading dataset...")

    response = requests.get(DATASET_URL, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to download. Status code: {response.status_code}")

    with open(SAVE_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download complete.")


if __name__ == "__main__":
    download_dataset()