import os
import urllib.request
import tarfile
import shutil

# Define main paths based on the project structure
DATA_URL = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz"
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
ARCHIVE_PATH = os.path.join(RAW_DIR, "EnglishImg.tgz")
EXTRACTED_DIR = os.path.join(RAW_DIR, "EnglishImg")

def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    if not os.path.exists(ARCHIVE_PATH):
        print(f"Downloading Chars74K Natural Images to {ARCHIVE_PATH}...")
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
        print("Download complete.")
    else:
        print("Archive already exists. Skipping download.")

    if not os.path.exists(EXTRACTED_DIR):
        print("Extracting archive. This might take a moment...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)
        print("Extraction complete.")
    else:
        print("Directory already extracted. Skipping extraction.")

# I commented out the previous function because it assumed the directory 
# was named "BadImg", but the dataset actually has a typo ("BadImag").
# def process_all_digits_old():
#     categories = ["GoodImg", "BadImg"]
#     ...

def process_all_digits():
    # We use a dictionary to map our clean prefix to the actual (typo'd) folder name in the dataset
    category_mapping = {
        "GoodImg": "GoodImg",
        "BadImg": "BadImag"  # Handling the typo in the dataset
    }
    
    # Clean up the processed directory first to ensure a fresh start
    if os.path.exists(PROCESSED_DIR):
        print(f"Cleaning up old '{PROCESSED_DIR}' directory to prevent duplicates...")
        shutil.rmtree(PROCESSED_DIR)
        
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Copying and renaming files from both Good and Bad categories...")
    
    for prefix, actual_folder_name in category_mapping.items():
        source_dir = os.path.join(RAW_DIR, "English", "Img", actual_folder_name, "Bmp")
        
        for i in range(1, 11):
            folder_name = f"Sample{i:03d}"
            src_folder = os.path.join(source_dir, folder_name)
            digit_str = str(i - 1)
            dest_folder = os.path.join(PROCESSED_DIR, digit_str)
            
            os.makedirs(dest_folder, exist_ok=True)
            
            if os.path.exists(src_folder):
                for file_name in os.listdir(src_folder):
                    src_file = os.path.join(src_folder, file_name)
                    if os.path.isfile(src_file):
                        # Use the clean prefix (GoodImg_ / BadImg_) for the new file name
                        new_file_name = f"{prefix}_{file_name}"
                        dest_file = os.path.join(dest_folder, new_file_name)
                        
                        if not os.path.exists(dest_file):
                            shutil.copy2(src_file, dest_file)
                
                print(f"Successfully processed {prefix} for digit '{digit_str}'")
            else:
                print(f"Warning: Source folder {src_folder} was not found.")

if __name__ == "__main__":
    print("Starting data preparation...")
    download_and_extract()
    process_all_digits()
    print("Data preparation finished successfully! Data is fresh and ready.")