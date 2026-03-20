import os
import tarfile
import urllib.request
import h5py
import cv2
import numpy as np

DATA_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/svhn'))
DATA_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/segmentation/natural'))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_name(f, idx):
    # Depending on the dimension of the dataset, it can be a reference or a value.
    try:
        name_ref = f['/digitStruct/name'][idx][0]
        # v[0] gets the value from the dataset the reference points to
        return ''.join([chr(v[0]) for v in f[name_ref][()]])
    except Exception as e:
        return None

def get_bbox(f, idx):
    item = f['/digitStruct/bbox'][idx].item()
    obj = f[item]
    
    def get_attr(attr):
        val = obj[attr]
        if len(val) > 1:
            return [f[val[i].item()][()][0][0] for i in range(len(val))]
        else:
            return [val[()][0][0]]
            
    boxes = zip(get_attr('left'), get_attr('top'), get_attr('width'), get_attr('height'))
    return list(boxes)

def download_and_extract():
    url = "https://ufldl.stanford.edu/housenumbers/test.tar.gz"
    tar_path = os.path.join(DATA_RAW, "test.tar.gz")
    if not os.path.exists(tar_path):
        print("Downloading SVHN test Format 1 (~276MB)...")
        urllib.request.urlretrieve(url, tar_path)
    
    test_dir = os.path.join(DATA_RAW, "test")
    if not os.path.exists(test_dir):
        print("Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=DATA_RAW)

def process_svhn_masks():
    print("Processing SVHN Format 1 into masks...")
    mat_file = os.path.join(DATA_RAW, "test", "digitStruct.mat")
    if not os.path.exists(mat_file): print("digitStruct.mat not found"); return
    
    ensure_dir(DATA_PROCESSED)
    
    with h5py.File(mat_file, 'r') as f:
        length = len(f['/digitStruct/name'])
        # Process first 500 images
        processed = 0
        for i in range(length):
            if processed >= 500: break
            name = get_name(f, i)
            if not name: continue
            try:
                bboxes = get_bbox(f, i)
            except Exception as e:
                continue
            
            img_path = os.path.join(DATA_RAW, "test", name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for box in bboxes:
                left, top, width, height = box
                left, top, width, height = int(left), int(top), int(width), int(height)
                # Cap the bounding boxes
                left = max(0, left)
                top = max(0, top)
                right = min(w, left+width)
                bottom = min(h, top+height)
                cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
                
            sample_dir = os.path.join(DATA_PROCESSED, str(processed))
            ensure_dir(sample_dir)
            cv2.imwrite(os.path.join(sample_dir, "image.jpg"), img)
            cv2.imwrite(os.path.join(sample_dir, "mask.png"), mask)
            processed += 1

if __name__ == "__main__":
    ensure_dir(DATA_RAW)
    ensure_dir(DATA_PROCESSED)
    download_and_extract()
    process_svhn_masks()
    print("SVHN processing complete. Files in `data/segmentation/natural`.")
