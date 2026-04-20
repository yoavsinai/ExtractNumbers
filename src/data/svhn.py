import os
import h5py
import json
import shutil
import kagglehub
from tqdm import tqdm
from PIL import Image

def get_name(n, f):
    return ''.join([chr(c[0]) for c in f[n[0]]])

def get_bbox(ref, f):
    bbox = {}
    attr = ['height', 'left', 'top', 'width', 'label']
    for a in attr:
        ds = f[ref][a]
        if ds.shape[0] > 1:
            bbox[a] = [f[ds[j][0]][0][0] for j in range(ds.shape[0])]
        else:
            bbox[a] = [ds[0][0]]
    return bbox

def parse_digit_struct(mat_file, limit=None):
    f = h5py.File(mat_file, 'r')
    names = f['digitStruct/name']
    bboxes = f['digitStruct/bbox']
    data = []
    total = len(names)
    if limit: total = min(total, limit)
    for i in tqdm(range(total), desc="Parsing SVHN Meta"):
        data.append({'name': get_name(names[i], f), 'bbox': get_bbox(bboxes[i][0], f)})
    f.close()
    return data

def prepare(output_base_dir, limit=None):
    print("--- Preparing SVHN Dataset ---")
    raw_path = kagglehub.dataset_download("stanfordu/street-view-house-numbers")
    
    mat_file = os.path.join(raw_path, "train_digitStruct.mat")
    if not os.path.exists(mat_file):
        for root, dirs, files in os.walk(raw_path):
            if "train_digitStruct.mat" in files:
                mat_file = os.path.join(root, "train_digitStruct.mat")
                break
    
    if not os.path.exists(mat_file):
        print("Error: Could not find train_digitStruct.mat")
        return

    ext_dir = os.path.join(os.path.dirname(mat_file), "train", "train")
    if not os.path.exists(ext_dir):
        ext_dir = os.path.dirname(mat_file)

    annotations = parse_digit_struct(mat_file, limit=limit)
    
    dataset_dir = os.path.join(output_base_dir, "svhn")
    os.makedirs(dataset_dir, exist_ok=True)

    for i, entry in enumerate(tqdm(annotations, desc="SVHN")):
        img_filename = entry['name']
        sample_id = os.path.splitext(img_filename)[0]
        img_src = os.path.join(ext_dir, img_filename)
        if not os.path.exists(img_src): continue
        
        sample_folder = os.path.join(dataset_dir, f"sample_{sample_id}")
        os.makedirs(sample_folder, exist_ok=True)
        img_dest = os.path.join(sample_folder, "original.png")
        shutil.copy(img_src, img_dest)
        
        with Image.open(img_dest) as img:
            w, h = img.size
            
        bbox_data = entry['bbox']
        num_digits = len(bbox_data['label'])
        digits_metadata = []
        full_value = ""
        x_coords, y_coords, max_x, max_y = [], [], [], []
        
        for j in range(num_digits):
            label = int(bbox_data['label'][j])
            if label == 10: label = 0
            y, x, wd, ht = float(bbox_data['top'][j]), float(bbox_data['left'][j]), float(bbox_data['width'][j]), float(bbox_data['height'][j])
            digits_metadata.append({"label": label, "bounding_box": {"x": x, "y": y, "width": wd, "height": ht}})
            full_value += str(label)
            x_coords.append(x); y_coords.append(y); max_x.append(x+wd); max_y.append(y+ht)
            
        full_box = {"x": min(x_coords), "y": min(y_coords), "width": max(max_x)-min(x_coords), "height": max(max_y)-min(y_coords)}
        
        metadata = {
            "image_metadata": {"sample_index": i, "filename": "original.png", "width": w, "height": h},
            "detected_numbers": [{"full_value": full_value, "full_bounding_box": full_box, "digits": digits_metadata}]
        }
        with open(os.path.join(sample_folder, "annotations.json"), "w") as f:
            json.dump(metadata, f, indent=4)
