import os
import json
import shutil
import kagglehub
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

def prepare(output_base_dir, limit=None):
    raw_path = kagglehub.dataset_download("trainingdatapro/ocr-race-numbers")
    
    xml_file = os.path.join(raw_path, "annotations.xml")
    if not os.path.exists(xml_file):
        print("Error: Could not find annotations.xml")
        return

    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    dataset_dir = os.path.join(output_base_dir, "race_numbers")
    os.makedirs(dataset_dir, exist_ok=True)

    images = root.findall('image')
    if limit: images = images[:limit]

    for i, img_elem in enumerate(tqdm(images, desc="RaceNumbers")):
        img_id = img_elem.get('id')
        img_name = img_elem.get('name') # e.g. "images/0.png"
        w = int(img_elem.get('width'))
        h = int(img_elem.get('height'))
        
        img_src = os.path.join(raw_path, img_name)
        if not os.path.exists(img_src):
            # Check if it's just the basename
            img_src = os.path.join(raw_path, "images", os.path.basename(img_name))
            if not os.path.exists(img_src):
                continue
        
        sample_folder = os.path.join(dataset_dir, f"sample_{img_id}")
        os.makedirs(sample_folder, exist_ok=True)
        img_dest = os.path.join(sample_folder, "original.png")
        shutil.copy(img_src, img_dest)
        
        numbers = []
        for box in img_elem.findall('box'):
            if box.get('label') == 'number':
                text = ""
                attr = box.find("./attribute[@name='text']")
                if attr is not None:
                    text = attr.text
                
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                numbers.append({
                    "full_value": text,
                    "full_bounding_box": {
                        "x": xtl,
                        "y": ytl,
                        "width": xbr - xtl,
                        "height": ybr - ytl
                    },
                    "digits": [] # This dataset doesn't provide individual digits
                })
        
        metadata = {
            "image_metadata": {
                "sample_index": i,
                "filename": "original.png",
                "width": w,
                "height": h
            },
            "detected_numbers": numbers
        }
        
        with open(os.path.join(sample_folder, "annotations.json"), "w") as f:
            json.dump(metadata, f, indent=4)
