import os
import xml.etree.ElementTree as ET
import json
import shutil
from tqdm import tqdm

def prepare(output_root, base_path=None, limit=None):
    if base_path is None:
        import kagglehub
        print("Downloading trains dataset through kagglehub...")
        base_path = kagglehub.dataset_download("trainingdatapro/ocr-trains-dataset")
    xml_path = os.path.join(base_path, "annotations.xml")
    images_dir = os.path.join(base_path, "images")
    
    print(f"Parsing XML from {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create a dedicated subfolder for this dataset
    dataset_dir = os.path.join(output_root, "ocr_trains")
    os.makedirs(dataset_dir, exist_ok=True)
    
    processed_count = 0
    for image_elem in tqdm(root.findall('image'), desc="Converting Trains Dataset"):
        if limit and processed_count >= limit:
            break
            
        img_id = image_elem.get('id')
        img_name = os.path.basename(image_elem.get('name'))
        src_img_path = os.path.join(base_path, "images", img_name)
        
        if not os.path.exists(src_img_path):
            continue
            
        global_boxes = []
        full_value = ""
        
        for box in image_elem.findall('box'):
            if box.get('label') == 'numbers':
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                global_boxes.append([xtl, ytl, xbr, ybr])
                
                # Get text attribute
                for attr in box.findall('attribute'):
                    if attr.get('name') == 'text':
                        full_value = attr.text
                        break
        
        # Create output directory
        sample_id = f"trains_{img_id}"
        sample_dir = os.path.join(dataset_dir, f"sample_{sample_id}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Copy image
        shutil.copy(src_img_path, os.path.join(sample_dir, "original.png"))
        
        # Create annotations.json in standard format
        anno_out = {
            "image_metadata": {
                "sample_id": sample_id,
                "width": int(image_elem.get('width')),
                "height": int(image_elem.get('height'))
            },
            "detected_numbers": []
        }
        
        for box_coords, full_val in zip(global_boxes, [full_value] * len(global_boxes)):
            anno_out["detected_numbers"].append({
                "full_value": full_val,
                "full_bounding_box": {
                    "x": float(box_coords[0]),
                    "y": float(box_coords[1]),
                    "width": float(box_coords[2] - box_coords[0]),
                    "height": float(box_coords[3] - box_coords[1])
                },
                "digits": []
            })
        
        with open(os.path.join(sample_dir, "annotations.json"), 'w') as f:
            json.dump(anno_out, f, indent=4)
            
        processed_count += 1
        
    print(f"Done! Processed {processed_count} images into {output_root}")

if __name__ == "__main__":
    import kagglehub
    print("Downloading trains dataset through kagglehub...")
    base = kagglehub.dataset_download("trainingdatapro/ocr-trains-dataset")
    out = os.path.join("data", "trains_dataset")
    prepare(base, out)
