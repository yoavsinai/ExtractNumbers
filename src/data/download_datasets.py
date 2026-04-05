import os
import kagglehub
import torchvision
import requests
import tarfile
import io

def download_svhn_extra(data_dir):
    """
    Downloads the SVHN test set bounding box information if not already present.
    """
    svhn_dir = os.path.join(data_dir, 'svhn')
    os.makedirs(svhn_dir, exist_ok=True)
    
    mat_file_path = os.path.join(svhn_dir, 'test', 'digitStruct.mat')
    
    if os.path.exists(mat_file_path):
        print("SVHN test bounding box file already exists.")
        return

    print("Downloading SVHN test set bounding box data (digitStruct.mat)...")
    url = "http://ufldl.stanford.edu/housenumbers/test.tar.gz"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Open the tarball from the downloaded content in memory
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            # Find the specific member and extract it
            for member in tar.getmembers():
                if member.name == 'test/digitStruct.mat':
                    tar.extract(member, path=svhn_dir)
                    print(f"Extracted '{member.name}' to '{svhn_dir}'")
                    # The file will be at svhn_dir/test/digitStruct.mat
                    return
        
        print("Error: 'test/digitStruct.mat' not found in the downloaded archive.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except tarfile.TarError as e:
        print(f"Error extracting tar file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def download_datasets():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True)
    
    print("Downloading SVHN...")
    torchvision.datasets.SVHN(root=data_dir, split='train', download=True)
    torchvision.datasets.SVHN(root=data_dir, split='test', download=True)
    
    # Download the missing bounding box file for the test set
    download_svhn_extra(data_dir)
    
    print("Downloading Handwritten Digits 0-9 from Kaggle...")
    try:
        path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
        print(f"Handwritten Digits downloaded to {path}")
    except Exception as e:
        print(f"Error downloading Handwritten Digits: {e}")

if __name__ == "__main__":
    download_datasets()
