import os
import kagglehub
import torchvision

def download_datasets():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    
    print("Downloading SVHN...")
    torchvision.datasets.SVHN(root=data_dir, split='train', download=True)
    torchvision.datasets.SVHN(root=data_dir, split='test', download=True)
    
    print("Downloading Handwritten Digits 0-9 from Kaggle...")
    try:
        path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
        print(f"Handwritten Digits downloaded to {path}")
    except Exception as e:
        print(f"Error downloading Handwritten Digits: {e}")

if __name__ == "__main__":
    download_datasets()
