import os
import shutil
import cv2
import json
import xml.etree.ElementTree as ET
import requests
import zipfile
from tqdm import tqdm
import re
import numpy as np

# Conversion status tracker
conversion_results = {
    "moving_mnist": {"success": False, "count": 0, "error": None},
    "dstext_v2": {"success": False, "count": 0, "error": None},
    "roadtext": {"success": False, "count": 0, "error": None},
    "bovtext": {"success": False, "count": 0, "error": None},
    "icdar_svt": {"success": False, "count": 0, "error": None},
}

def parse_roadtext_json_tolerant(file_path):
    """
    Parse RoadText JSON file that may be partially corrupted.
    Attempts to extract as many complete sequence entries as possible.
    """
    print(f"[*] Attempting tolerant parsing of {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[!] Could not read file: {e}")
        return {}
    
    data = {}
    
    # Find all sequence entries with pattern: "NNNN": {...}
    # Use a simpler approach: find sequence IDs and try to extract their data
    pattern = r'"(\d+)"\s*:\s*\{'
    matches = list(re.finditer(pattern, content))
    
    print(f"[*] Found {len(matches)} sequence entry markers")
    
    for idx, match in enumerate(matches):
        seq_id = match.group(1)
        start_pos = match.start()
        
        # Try to find the end of this sequence by counting braces
        brace_count = 0
        in_string = False
        escape_next = False
        end_pos = -1
        
        for pos in range(match.end(), min(len(content), match.end() + 100000)):
            char = content[pos]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = pos + 1
                        break
        
        if end_pos > 0:
            # Try to parse this sequence
            seq_content = content[start_pos:end_pos]
            try:
                # Wrap in braces to make it valid JSON
                wrapped = '{' + seq_content + '}'
                seq_data = json.loads(wrapped)
                data[seq_id] = seq_data[seq_id]
                if (idx + 1) % 1000 == 0:
                    print(f"  [✓] Extracted {idx + 1} sequences...")
            except:
                # Skip this sequence if it can't be parsed
                pass
    
    print(f"[✓] Successfully extracted {len(data)} sequences from corrupted JSON")
    return data


def convert_moving_mnist_from_npy(npy_path, dest_dir, max_sequences=None):
    """
    Convert Moving MNIST from NPY file directly to MP4 videos with annotations.
    Loads shape (20, 10000, 64, 64) → (frames, sequences, height, width)
    Processes each sequence: write frames → MP4, detect digits → JSON annotations
    """
    print("\n" + "="*60)
    print("Converting Moving MNIST from NPY to unified format...")
    print("="*60)
    
    dataset_name = "moving_mnist"
    
    if not os.path.exists(npy_path):
        error_msg = f"NPY file not found: {npy_path}"
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False
    
    try:
        print(f"[*] Loading NPY file: {npy_path}")
        # Try regular load first
        data = None
        load_error = None
        try:
            data = np.load(npy_path, allow_pickle=False)
            print(f"[✓] Standard load successful. Shape: {data.shape}")
        except Exception as e:
            load_error = str(e)
            print(f"[!] Standard load failed: {load_error}")
            print(f"[*] Attempting fallback: read raw NPY file and infer shape...")
            
            try:
                # Read raw NPY file and try to determine shape
                with open(npy_path, 'rb') as f:
                    import struct
                    # NPY format: magic (6) + version (2) + header_len (4) + header
                    magic = f.read(6)
                    if magic != b'\x93NUMPY':
                        raise ValueError("Not a valid NPY file")
                    
                    version_major, version_minor = struct.unpack('BB', f.read(2))
                    print(f"[*] NPY version: {version_major}.{version_minor}")
                    
                    if version_major == 1:
                        header_len = struct.unpack('<I', f.read(4))[0]
                    else:
                        header_len = struct.unpack('<Q', f.read(8))[0]
                    
                    header = f.read(header_len).decode('latin1')
                    print(f"[*] NPY header info (first 200 chars): {header[:200]}")
                    
                    # Skip rest of header padding and read data
                    data_start = f.tell()
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()
                    data_size = file_size - data_start
                    
                    print(f"[*] Data size: {data_size} bytes ({data_size / 1e6:.1f} MB)")
                    
                    # Read as raw uint8
                    f.seek(data_start)
                    raw_data = np.frombuffer(f.read(), dtype=np.uint8)
                    print(f"[*] Raw data elements: {raw_data.size:,}")
                    
                    # Try to reshape to 3D (num_images, 64, 64)
                    if raw_data.size % (64 * 64) == 0:
                        num_images = raw_data.size // (64 * 64)
                        data = raw_data.reshape(num_images, 64, 64)
                        print(f"[✓] Reshaped to: ({num_images}, 64, 64)")
                    else:
                        raise ValueError(f"Cannot reshape {raw_data.size} bytes into 64×64 images")
                        
            except Exception as e2:
                print(f"[!] Fallback also failed: {e2}")
                error_msg = f"Could not load NPY file: {load_error}"
                conversion_results[dataset_name]["error"] = error_msg
                return False
        
        print(f"[*] Final data shape: {data.shape}, dtype: {data.dtype}")
        
        # Determine dimension interpretation
        if data.ndim == 4 and data.shape[0] == 20:
            # Standard (frames, sequences, height, width)
            num_frames, num_sequences, height, width = data.shape
            print(f"[✓] Detected (frames, sequences, h, w): {num_frames}×{num_sequences}×{height}×{width}")
        elif data.ndim == 4 and data.shape[1] == 20:
            # Swapped (sequences, frames, height, width) - transpose
            print("[*] Detected (sequences, frames, h, w) - transposing to (frames, sequences, h, w)")
            data = data.transpose(1, 0, 2, 3)
            num_frames, num_sequences, height, width = data.shape
        elif data.ndim == 3:
            # (num_images, height, width) - add frame dimension
            print("[*] Detected (images, h, w) - treating each image as single-frame sequence")
            num_sequences, height, width = data.shape
            num_frames = 1
            data = data.reshape(num_sequences, 1, height, width)
        elif data.ndim == 4:
            # Some other 4D arrangement
            print(f"[!] Warning: 4D data with unexpected shape {data.shape}")
            # Try to guess: assume last two dims are 64×64
            if data.shape[-2:] == (64, 64):
                # Reshape to (sequences, frames, 64, 64)
                data = data.reshape(-1, 1, 64, 64)
                num_sequences, num_frames, height, width = data.shape
                print(f"[*] Reshaped to: ({num_sequences}, {num_frames}, 64, 64)")
            else:
                error_msg = f"Cannot interpret shape: {data.shape}"
                print(f"[!] {error_msg}")
                conversion_results[dataset_name]["error"] = error_msg
                return False
        else:
            error_msg = f"Expected 3D or 4D array, got {data.ndim}D with shape {data.shape}"
            print(f"[!] {error_msg}")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        print(f"[✓] Loaded data: {num_frames} frames × {num_sequences} sequences × {height}×{width}")
        
        if max_sequences:
            num_sequences = min(max_sequences, num_sequences)
            print(f"[*] Limiting to {num_sequences} sequences")
        
        os.makedirs(dest_dir, exist_ok=True)
        
        successful_count = 0
        failed_sequences = []
        
        for seq_idx in tqdm(range(num_sequences), desc="Moving MNIST Sequences"):
            try:
                seq_name = f"sequence_{seq_idx}"
                seq_dest_path = os.path.join(dest_dir, seq_name)
                os.makedirs(seq_dest_path, exist_ok=True)
                
                video_dest_path = os.path.join(seq_dest_path, "video.mp4")
                anno_dest_path = os.path.join(seq_dest_path, "annotations.json")
                
                # 1. Write frames to MP4 video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_dest_path, fourcc, 10.0, (width, height))
                
                if not out.isOpened():
                    failed_sequences.append((seq_name, "Could not create video writer"))
                    continue
                
                frames_written = 0
                frame_annotations = {}
                
                for frame_idx in range(num_frames):
                    # Load frame data (grayscale 64×64)
                    frame_data = data[frame_idx, seq_idx]
                    
                    # Convert to 8-bit if needed
                    if frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8) if frame_data.max() <= 1 else frame_data.astype(np.uint8)
                    
                    # Convert grayscale to BGR for video writing
                    frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                    out.write(frame_bgr)
                    frames_written += 1
                    
                    # 2. Detect digits via threshold + contours (same as download script)
                    _, thresh = cv2.threshold(frame_data, 20, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    digits = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Filter out tiny noise contours
                        if w > 3 and h > 3:
                            digits.append({
                                "label": "-1",
                                "bounding_box": {
                                    "x": float(x),
                                    "y": float(y),
                                    "width": float(w),
                                    "height": float(h)
                                }
                            })
                    
                    if digits:
                        # Compute bounding box for all digits
                        xs = [d["bounding_box"]["x"] for d in digits]
                        ys = [d["bounding_box"]["y"] for d in digits]
                        x_max = max([d["bounding_box"]["x"] + d["bounding_box"]["width"] for d in digits])
                        y_max = max([d["bounding_box"]["y"] + d["bounding_box"]["height"] for d in digits])
                        
                        frame_annotations[str(frame_idx)] = {
                            "detected_numbers": [{
                                "full_value": "digit_seq",
                                "full_bounding_box": {
                                    "x": float(min(xs)),
                                    "y": float(min(ys)),
                                    "width": float(x_max - min(xs)),
                                    "height": float(y_max - min(ys))
                                },
                                "digits": digits
                            }]
                        }
                
                out.release()
                
                if frames_written == 0:
                    failed_sequences.append((seq_name, "No frames were written to video"))
                    continue
                
                # 3. Write annotations
                anno_data = {
                    "video_metadata": {
                        "sample_id": f"moving_mnist/{seq_name}",
                        "width": width,
                        "height": height,
                        "fps": 10.0,
                        "frames_written": frames_written
                    },
                    "frames": frame_annotations
                }
                
                with open(anno_dest_path, 'w') as f:
                    json.dump(anno_data, f, indent=2)
                
                successful_count += 1
                
            except Exception as e:
                failed_sequences.append((seq_name, str(e)))
        
        if successful_count == 0:
            error_msg = f"No sequences converted successfully"
            print(f"[!] {error_msg}")
            if failed_sequences:
                for seq, err in failed_sequences[:5]:
                    print(f"  - {seq}: {err}")
                if len(failed_sequences) > 5:
                    print(f"  ... and {len(failed_sequences) - 5} more")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        conversion_results[dataset_name]["success"] = True
        conversion_results[dataset_name]["count"] = successful_count
        print(f"\n[✓] Moving MNIST: {successful_count}/{num_sequences} sequences converted successfully")
        
        if failed_sequences:
            print(f"[!] {len(failed_sequences)} sequences failed:")
            for seq, err in failed_sequences[:5]:
                print(f"  - {seq}: {err}")
            if len(failed_sequences) > 5:
                print(f"  ... and {len(failed_sequences) - 5} more")
        
        return True
        
    except Exception as e:
        error_msg = f"Error during conversion: {e}"
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False


def convert_moving_mnist(src_dir, dest_dir):
    print("\n" + "="*60)
    print("Converting Moving MNIST to unified format...")
    print("="*60)
    
    dataset_name = "moving_mnist"
    
    if not os.path.exists(src_dir):
        error_msg = f"Source directory {src_dir} does not exist."
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False
    
    try:
        sequences = [d for d in os.listdir(src_dir) if d.startswith("sequence_") and os.path.isdir(os.path.join(src_dir, d))]
        
        if not sequences:
            error_msg = f"No sequences found in {src_dir} (looking for folders starting with 'sequence_')"
            print(f"[!] {error_msg}")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        print(f"[✓] Found {len(sequences)} sequences to process")
        os.makedirs(dest_dir, exist_ok=True)
        
        successful_count = 0
        failed_sequences = []
        
        for seq_name in tqdm(sequences, desc="Moving MNIST Sequences"):
            try:
                seq_src_path = os.path.join(src_dir, seq_name)
                seq_dest_path = os.path.join(dest_dir, seq_name)
                os.makedirs(seq_dest_path, exist_ok=True)
                
                video_dest_path = os.path.join(seq_dest_path, "video.mp4")
                anno_dest_path = os.path.join(seq_dest_path, "annotations.json")
                
                # 1. Compile PNG frames to MP4
                frame_files = sorted([f for f in os.listdir(seq_src_path) if f.startswith("frame_") and f.endswith(".png")],
                                     key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))
                
                if not frame_files:
                    failed_sequences.append((seq_name, "No PNG frames found"))
                    continue
                
                # Read first frame and validate
                first_frame_path = os.path.join(seq_src_path, frame_files[0])
                first_frame = cv2.imread(first_frame_path)
                
                if first_frame is None:
                    failed_sequences.append((seq_name, f"Could not read first frame: {frame_files[0]}"))
                    continue
                
                height, width, _ = first_frame.shape
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_dest_path, fourcc, 10.0, (width, height))
                
                if not out.isOpened():
                    failed_sequences.append((seq_name, "Could not create video writer"))
                    continue
                
                frames_written = 0
                for f_file in frame_files:
                    frame_img = cv2.imread(os.path.join(seq_src_path, f_file))
                    if frame_img is None:
                        print(f"  [!] Warning: Could not read frame {f_file} in {seq_name}, skipping")
                        continue
                    out.write(frame_img)
                    frames_written += 1
                
                out.release()
                
                if frames_written == 0:
                    failed_sequences.append((seq_name, "No frames were successfully written to video"))
                    continue
                
                # 2. Parse XML GT annotations
                xml_file = f"{seq_name}_GT.xml"
                xml_path = os.path.join(seq_src_path, xml_file)
                
                anno_data = {
                    "video_metadata": {
                        "sample_id": f"moving_mnist/{seq_name}",
                        "width": width,
                        "height": height,
                        "fps": 10.0,
                        "frames_written": frames_written
                    },
                    "frames": {}
                }
                
                if os.path.exists(xml_path):
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        
                        for frame in root.findall('frame'):
                            frame_id = int(frame.get('ID')) - 1  # 0-indexed conversion
                            detected_numbers = []
                            
                            boxes = []
                            digits = []
                            for obj in frame.findall('object'):
                                points = obj.findall('Point')
                                if len(points) >= 4:
                                    xs = [int(pt.get('x')) for pt in points]
                                    ys = [int(pt.get('y')) for pt in points]
                                    x1, y1 = min(xs), min(ys)
                                    x2, y2 = max(xs), max(ys)
                                    w = x2 - x1
                                    h = y2 - y1
                                    
                                    boxes.append({
                                        "x": float(x1),
                                        "y": float(y1),
                                        "width": float(w),
                                        "height": float(h)
                                    })
                                    digits.append({
                                        "label": "digit",
                                        "bounding_box": {
                                            "x": float(x1),
                                            "y": float(y1),
                                            "width": float(w),
                                            "height": float(h)
                                        }
                                    })
                                    
                            if boxes:
                                xs = [b["x"] for b in boxes]
                                ys = [b["y"] for b in boxes]
                                x_min = min(xs)
                                y_min = min(ys)
                                x_max = max([b["x"] + b["width"] for b in boxes])
                                y_max = max([b["y"] + b["height"] for b in boxes])
                                
                                detected_numbers.append({
                                    "full_value": "digit_seq",
                                    "full_bounding_box": {
                                        "x": float(x_min),
                                        "y": float(y_min),
                                        "width": float(x_max - x_min),
                                        "height": float(y_max - y_min)
                                    },
                                    "digits": digits
                                })
                                
                            if detected_numbers:
                                anno_data["frames"][str(frame_id)] = {
                                    "detected_numbers": detected_numbers
                                }
                    except Exception as e:
                        print(f"  [!] Warning: Error parsing XML for {seq_name}: {e}")
                else:
                    print(f"  [!] Warning: XML annotations not found for {seq_name}")
                
                with open(anno_dest_path, 'w') as f:
                    json.dump(anno_data, f, indent=4)
                
                successful_count += 1
                
            except Exception as e:
                failed_sequences.append((seq_name, str(e)))
                print(f"  [!] Error processing {seq_name}: {e}")
        
        # Report results for this dataset
        conversion_results[dataset_name]["count"] = successful_count
        conversion_results[dataset_name]["success"] = successful_count > 0
        
        print(f"\n[✓] Moving MNIST: {successful_count}/{len(sequences)} sequences converted successfully")
        if failed_sequences:
            print(f"[!] Failed sequences ({len(failed_sequences)}):")
            for seq, reason in failed_sequences[:5]:  # Show first 5 failures
                print(f"    - {seq}: {reason}")
            if len(failed_sequences) > 5:
                print(f"    ... and {len(failed_sequences) - 5} more")
        
        return successful_count > 0
        
    except Exception as e:
        error_msg = f"Unexpected error in Moving MNIST conversion: {e}"
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False


def convert_dstext_v2(src_dir, dest_dir):
    print("\n" + "="*60)
    print("Converting DSText V2 to unified format...")
    print("="*60)
    
    dataset_name = "dstext_v2"
    
    if not os.path.exists(src_dir):
        error_msg = f"Source directory {src_dir} does not exist."
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False
    
    try:
        # Check if there are any XML files in src_dir
        has_xml = False
        for root, dirs, files in os.walk(src_dir):
            if any(f.endswith(".xml") for f in files):
                has_xml = True
                break
        
        if not has_xml:
            print("[!] XML annotations not found under DSText_V2 folder.")
            print("[*] Attempting to download annotations archive from Zenodo (record 10010840)...")
            ann_url = "https://zenodo.org/records/10010840/files/V2_Ann_Train.zip?download=1"
            ann_zip = os.path.join(src_dir, "V2_Ann_Train.zip")
            try:
                r = requests.get(ann_url, stream=True, timeout=30)
                r.raise_for_status()
                with open(ann_zip, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("[✓] Download complete. Extracting annotations...")
                with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
                    zip_ref.extractall(src_dir)
                os.remove(ann_zip)
                print("[✓] Annotations extracted successfully!")
            except Exception as e:
                error_msg = f"Failed to download or extract annotations: {e}"
                print(f"[!] {error_msg}")
                conversion_results[dataset_name]["error"] = error_msg
                return False
        
        found_videos = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".mp4"):
                    found_videos.append((root, file))
        
        if not found_videos:
            error_msg = "No MP4 video files found in DSText_V2"
            print(f"[!] {error_msg}")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        print(f"[✓] Found {len(found_videos)} videos to process")
        os.makedirs(dest_dir, exist_ok=True)
        
        successful_count = 0
        failed_videos = []
        
        for root, file in tqdm(found_videos, desc="DSText V2 Videos"):
            try:
                video_src_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                
                # Associated xml file
                xml_name = f"{video_name}_GT.xml"
                xml_path = os.path.join(root, xml_name)
                if not os.path.exists(xml_path):
                    # Try finding any XML in the same folder
                    xml_files = [f for f in os.listdir(root) if f.endswith(".xml")]
                    if xml_files:
                        xml_path = os.path.join(root, xml_files[0])
                
                if not os.path.exists(xml_path):
                    failed_videos.append((video_name, "No XML annotation file found"))
                    continue
                
                sample_folder = os.path.join(dest_dir, video_name)
                os.makedirs(sample_folder, exist_ok=True)
                
                video_dest_path = os.path.join(sample_folder, "video.mp4")
                anno_dest_path = os.path.join(sample_folder, "annotations.json")
                
                # Copy video file
                shutil.copy(video_src_path, video_dest_path)
                
                # Parse video metadata
                cap = cv2.VideoCapture(video_dest_path)
                if not cap.isOpened():
                    failed_videos.append((video_name, "Could not read video file"))
                    continue
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                anno_data = {
                    "video_metadata": {
                        "sample_id": f"dstext_v2/{video_name}",
                        "width": width,
                        "height": height,
                        "fps": fps
                    },
                    "frames": {}
                }
                
                try:
                    tree = ET.parse(xml_path)
                    xml_root = tree.getroot()
                    
                    for frame in xml_root.findall('frame'):
                        frame_id = int(frame.get('ID')) - 1  # 0-indexed
                        detected_numbers = []
                        
                        for obj in frame.findall('object'):
                            transcription = obj.get('Transcription', '')
                            if not transcription or transcription == "###":
                                continue
                            
                            points = obj.findall('Point')
                            if len(points) >= 4:
                                xs = [int(pt.get('x')) for pt in points]
                                ys = [int(pt.get('y')) for pt in points]
                                x1, y1 = min(xs), min(ys)
                                x2, y2 = max(xs), max(ys)
                                w = x2 - x1
                                h = y2 - y1
                                
                                gx = float(x1)
                                gy = float(y1)
                                gw = float(w)
                                gh = float(h)
                                
                                # Clean transcription for digit extraction
                                cleaned_trans = "".join([c for c in transcription if c.isdigit()])
                                digits = []
                                if cleaned_trans:
                                    dw = gw / len(cleaned_trans)
                                    for idx, char in enumerate(cleaned_trans):
                                        digits.append({
                                            "label": int(char),
                                            "bounding_box": {
                                                "x": float(gx + idx * dw),
                                                "y": float(gy),
                                                "width": float(dw),
                                                "height": float(gh)
                                            }
                                        })
                                
                                detected_numbers.append({
                                    "full_value": cleaned_trans if cleaned_trans else transcription,
                                    "full_bounding_box": {
                                        "x": gx,
                                        "y": gy,
                                        "width": gw,
                                        "height": gh
                                    },
                                    "digits": digits
                                })
                        
                        if detected_numbers:
                            anno_data["frames"][str(frame_id)] = {
                                "detected_numbers": detected_numbers
                            }
                except Exception as e:
                    print(f"  [!] Warning: Error parsing XML for {video_name}: {e}")
                
                with open(anno_dest_path, 'w') as f:
                    json.dump(anno_data, f, indent=4)
                
                successful_count += 1
                
            except Exception as e:
                failed_videos.append((video_name, str(e)))
                print(f"  [!] Error processing {video_name}: {e}")
        
        # Report results for this dataset
        conversion_results[dataset_name]["count"] = successful_count
        conversion_results[dataset_name]["success"] = successful_count > 0
        
        print(f"\n[✓] DSText V2: {successful_count}/{len(found_videos)} videos converted successfully")
        if failed_videos:
            print(f"[!] Failed videos ({len(failed_videos)}):")
            for vid, reason in failed_videos[:5]:
                print(f"    - {vid}: {reason}")
            if len(failed_videos) > 5:
                print(f"    ... and {len(failed_videos) - 5} more")
        
        return successful_count > 0
        
    except Exception as e:
        error_msg = f"Unexpected error in DSText V2 conversion: {e}"
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False



# ============================================================
# ROADTEXT CONVERSION DISABLED (corrupted source annotations)
# ============================================================
# def convert_roadtext(src_dir, dest_dir):
#     print("\n" + "="*60)
#     print("Converting RoadText to unified format...")
#     print("="*60)
#     
#     dataset_name = "roadtext"
#     
#     if not os.path.exists(src_dir):
#         error_msg = f"Source directory {src_dir} does not exist."
#         print(f"[!] {error_msg}")
#         conversion_results[dataset_name]["error"] = error_msg
#         return False
#     
#     try:
#         annotation_path = os.path.join(src_dir, "roadtext-annotation-fixed.json")
#         if not os.path.exists(annotation_path):
#             error_msg = "Master annotation file 'roadtext-annotation-fixed.json' not found."
#             print(f"[!] {error_msg}")
#             conversion_results[dataset_name]["error"] = error_msg
#             return False
#         
#         data = None
#         with open(annotation_path, 'r') as f:
#             try:
#                 data = json.load(f)
#             except json.decoder.JSONDecodeError as e:
#                 print(f"[!] Warning: Annotation JSON is corrupted: {e}")
#                 print(f"[*] Attempting tolerant parsing to extract what we can...")
#                 data = parse_roadtext_json_tolerant(annotation_path)
#                 
#                 if not data:
#                     print(f"[!] Could not extract any sequences from corrupted JSON")
#                     conversion_results[dataset_name]["error"] = "JSON file corrupted and could not be parsed"
#                     return False
#         
#         # Find all mp4 files
#         found_videos = []
#         for root, dirs, files in os.walk(src_dir):
#             for file in files:
#                 if file.endswith(".mp4"):
#                     found_videos.append((root, file))
#         
#         if not found_videos:
#             error_msg = "No MP4 video files found in RoadText"
#             print(f"[!] {error_msg}")
#             conversion_results[dataset_name]["error"] = error_msg
#             return False
#         
#         print(f"[✓] Found {len(found_videos)} videos and annotation file")
#         os.makedirs(dest_dir, exist_ok=True)
#         
#         successful_count = 0
#         failed_videos = []
#         
#         for root, file in tqdm(found_videos, desc="RoadText Videos"):
#             try:
#                 video_src_path = os.path.join(root, file)
#                 seq_id = os.path.splitext(file)[0]
#                 
#                 # Check if we have annotations for this sequence ID
#                 if seq_id not in data:
#                     # Maybe the sequence ID in JSON is an integer or formatted differently
#                     matching_key = None
#                     for key in data.keys():
#                         if str(key) == str(seq_id) or str(key).endswith(str(seq_id)) or str(seq_id).endswith(str(key)):
#                             matching_key = key
#                             break
#                     if matching_key:
#                         seq_data = data[matching_key]
#                     else:
#                         failed_videos.append((seq_id, "No matching annotation in JSON file"))
#                         continue
#                 else:
#                     seq_data = data[seq_id]
#                 
#                 sample_folder = os.path.join(dest_dir, f"sequence_{seq_id}")
#                 os.makedirs(sample_folder, exist_ok=True)
#                 
#                 video_dest_path = os.path.join(sample_folder, "video.mp4")
#                 anno_dest_path = os.path.join(sample_folder, "annotations.json")
#                 
#                 shutil.copy(video_src_path, video_dest_path)
#                 
#                 cap = cv2.VideoCapture(video_dest_path)
#                 if not cap.isOpened():
#                     failed_videos.append((seq_id, "Could not read video file"))
#                     continue
#                 
#                 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 fps = float(cap.get(cv2.CAP_PROP_FPS))
#                 cap.release()
#                 
#                 anno_out = {
#                     "video_metadata": {
#                         "sample_id": f"roadtext/sequence_{seq_id}",
#                         "width": width,
#                         "height": height,
#                         "fps": fps
#                     },
#                     "frames": {}
#                 }
#                 
#                 for frame_key, frame_content in seq_data.items():
#                     try:
#                         frame_id = int(frame_key) - 1  # 0-indexed conversion
#                         detected_numbers = []
#                         
#                         if frame_content and isinstance(frame_content, dict):
#                             for label in frame_content.get("labels", []) or []:
#                                 box = label.get("box2d")
#                                 if box is None:
#                                     continue
#                                 ocr_text = label.get("ocr") or ""
#                                 cleaned_ocr = "".join([c for c in ocr_text if c.isdigit()])
#                                 
#                                 if not cleaned_ocr:
#                                     continue
#                                 
#                                 x1 = float(box.get("x1", 0))
#                                 y1 = float(box.get("y1", 0))
#                                 x2 = float(box.get("x2", 0))
#                                 y2 = float(box.get("y2", 0))
#                                 
#                                 gx = x1
#                                 gy = y1
#                                 gw = x2 - x1
#                                 gh = y2 - y1
#                                 
#                                 digits = []
#                                 if gw > 0:  # Avoid division by zero
#                                     dw = gw / len(cleaned_ocr)
#                                     for idx, char in enumerate(cleaned_ocr):
#                                         digits.append({
#                                             "label": int(char),
#                                             "bounding_box": {
#                                                 "x": float(gx + idx * dw),
#                                                 "y": float(gy),
#                                                 "width": float(dw),
#                                                 "height": float(gh)
#                                             }
#                                         })
#                                 
#                                 detected_numbers.append({
#                                     "full_value": cleaned_ocr,
#                                     "full_bounding_box": {
#                                         "x": gx,
#                                         "y": gy,
#                                         "width": gw,
#                                         "height": gh
#                                     },
#                                     "digits": digits
#                                 })
#                         
#                         if detected_numbers:
#                             anno_out["frames"][str(frame_id)] = {
#                                 "detected_numbers": detected_numbers
#                             }
#                     except Exception as e:
#                         print(f"  [!] Warning: Error parsing frame {frame_key} in {seq_id}: {e}")
#                 
#                 with open(anno_dest_path, 'w') as f:
#                     json.dump(anno_out, f, indent=4)
#                 
#                 successful_count += 1
#                 
#             except Exception as e:
#                 failed_videos.append((seq_id, str(e)))
#                 print(f"  [!] Error processing {seq_id}: {e}")
#         
#         # Report results for this dataset
#         conversion_results[dataset_name]["count"] = successful_count
#         conversion_results[dataset_name]["success"] = successful_count > 0
#         
#         print(f"\n[✓] RoadText: {successful_count}/{len(found_videos)} videos converted successfully")
#         if failed_videos:
#             print(f"[!] Failed videos ({len(failed_videos)}):")
#             for vid, reason in failed_videos[:5]:
#                 print(f"    - {vid}: {reason}")
#             if len(failed_videos) > 5:
#                 print(f"    ... and {len(failed_videos) - 5} more")
#         
#         return successful_count > 0
#         
#     except Exception as e:
#         error_msg = f"Unexpected error in RoadText conversion: {e}"
#         print(f"[!] {error_msg}")
#         conversion_results[dataset_name]["error"] = error_msg
#         return False



def convert_bovtext_or_icdar(src_dir, dest_dir, dataset_name):
    print("\n" + "="*60)
    print(f"Converting {dataset_name.upper()} to unified format...")
    print("="*60)
    
    if not os.path.exists(src_dir):
        error_msg = f"Source directory {src_dir} does not exist."
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False
    
    try:
        videos_dir = os.path.join(src_dir, "videos")
        annos_dir = os.path.join(src_dir, "annotations")
        
        if not os.path.exists(videos_dir):
            error_msg = f"Videos directory not found: {videos_dir}"
            print(f"[!] {error_msg}")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        if not os.path.exists(annos_dir):
            error_msg = f"Annotations directory not found: {annos_dir}"
            print(f"[!] {error_msg}")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        anno_files = [f for f in os.listdir(annos_dir) if f.endswith(".json")]
        
        if not anno_files:
            error_msg = f"No JSON annotation files found in {annos_dir}"
            print(f"[!] {error_msg}")
            conversion_results[dataset_name]["error"] = error_msg
            return False
        
        print(f"[✓] Found {len(anno_files)} annotation files")
        os.makedirs(dest_dir, exist_ok=True)
        
        successful_count = 0
        failed_videos = []
        
        for anno_file in tqdm(anno_files, desc=f"{dataset_name.upper()} Videos"):
            try:
                anno_path = os.path.join(annos_dir, anno_file)
                
                with open(anno_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        failed_videos.append((anno_file, f"Invalid JSON: {e}"))
                        continue
                
                video_name = data.get("video_name", "")
                if not video_name:
                    video_name = os.path.splitext(anno_file)[0].replace("_gt", "")
                
                video_file = f"{video_name}.mp4"
                video_src_path = os.path.join(videos_dir, video_file)
                
                if not os.path.exists(video_src_path):
                    # Try any video ending with .mp4
                    mp4_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
                    if mp4_files:
                        video_src_path = os.path.join(videos_dir, mp4_files[0])
                    else:
                        failed_videos.append((video_name, "No MP4 video file found"))
                        continue
                
                if not os.path.exists(video_src_path):
                    failed_videos.append((video_name, "Video file not found"))
                    continue
                
                sample_folder = os.path.join(dest_dir, video_name)
                os.makedirs(sample_folder, exist_ok=True)
                
                video_dest_path = os.path.join(sample_folder, "video.mp4")
                anno_dest_path = os.path.join(sample_folder, "annotations.json")
                
                shutil.copy(video_src_path, video_dest_path)
                
                cap = cv2.VideoCapture(video_dest_path)
                if not cap.isOpened():
                    failed_videos.append((video_name, "Could not read video file"))
                    continue
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                anno_out = {
                    "video_metadata": {
                        "sample_id": f"{dataset_name}/{video_name}",
                        "width": width,
                        "height": height,
                        "fps": fps
                    },
                    "frames": {}
                }
                
                for frame_data in data.get("frames", []):
                    try:
                        frame_idx = frame_data.get("frame_index")
                        detected_numbers = []
                        
                        for anno in frame_data.get("annotations", []):
                            box = anno.get("box", [0, 0, 0, 0])
                            text = anno.get("text") or ""
                            cleaned_trans = "".join([c for c in text if c.isdigit()])
                            
                            if not cleaned_trans:
                                continue
                            
                            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                            gx, gy = float(x1), float(y1)
                            gw, gh = float(x2 - x1), float(y2 - y1)
                            
                            digits = []
                            if gw > 0:  # Avoid division by zero
                                dw = gw / len(cleaned_trans)
                                for idx, char in enumerate(cleaned_trans):
                                    digits.append({
                                        "label": int(char),
                                        "bounding_box": {
                                            "x": float(gx + idx * dw),
                                            "y": float(gy),
                                            "width": float(dw),
                                            "height": float(gh)
                                        }
                                    })
                            
                            detected_numbers.append({
                                "full_value": cleaned_trans,
                                "full_bounding_box": {
                                    "x": gx,
                                    "y": gy,
                                    "width": gw,
                                    "height": gh
                                },
                                "digits": digits
                            })
                        
                        if detected_numbers:
                            anno_out["frames"][str(frame_idx)] = {
                                "detected_numbers": detected_numbers
                            }
                    except Exception as e:
                        print(f"  [!] Warning: Error parsing frame in {video_name}: {e}")
                
                with open(anno_dest_path, 'w') as f:
                    json.dump(anno_out, f, indent=4)
                
                successful_count += 1
                
            except Exception as e:
                failed_videos.append((video_name, str(e)))
                print(f"  [!] Error processing {video_name}: {e}")
        
        # Report results for this dataset
        conversion_results[dataset_name]["count"] = successful_count
        conversion_results[dataset_name]["success"] = successful_count > 0
        
        print(f"\n[✓] {dataset_name.upper()}: {successful_count}/{len(anno_files)} videos converted successfully")
        if failed_videos:
            print(f"[!] Failed videos ({len(failed_videos)}):")
            for vid, reason in failed_videos[:5]:
                print(f"    - {vid}: {reason}")
            if len(failed_videos) > 5:
                print(f"    ... and {len(failed_videos) - 5} more")
        
        return successful_count > 0
        
    except Exception as e:
        error_msg = f"Unexpected error in {dataset_name.upper()} conversion: {e}"
        print(f"[!] {error_msg}")
        conversion_results[dataset_name]["error"] = error_msg
        return False

def print_conversion_summary():
    """Print a comprehensive summary of all conversions."""
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    total_datasets = len(conversion_results)
    successful_datasets = sum(1 for r in conversion_results.values() if r["success"])
    total_videos = sum(r["count"] for r in conversion_results.values())
    
    for dataset, result in conversion_results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        count = result["count"]
        print(f"{status:12} | {dataset:15} | {count:3} videos converted")
        if result["error"] and not result["success"]:
            print(f"              | Error: {result['error']}")
    
    print("="*60)
    print(f"Total: {successful_datasets}/{total_datasets} datasets successful | {total_videos} total videos converted")
    print("="*60)
    
    return successful_datasets > 0

def main():
    print("\n" + "="*80)
    print(" VIDEO DATA CONVERSION PIPELINE")
    print("="*80)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_data_root = os.path.join(base_dir, "data")
    dest_data_root = os.path.join(base_dir, "data", "video_data")
    
    os.makedirs(dest_data_root, exist_ok=True)
    
    # Clean previous video_data to ensure fresh conversion
    # Skip mock_video so we don't have to regenerate it
    print("\n[*] Cleaning previous conversion results...")
    for d in os.listdir(dest_data_root):
        if d != "mock_video":
            dir_to_clean = os.path.join(dest_data_root, d)
            if os.path.isdir(dir_to_clean):
                try:
                    shutil.rmtree(dir_to_clean)
                    print(f"  [✓] Removed {d}/")
                except Exception as e:
                    print(f"  [!] Could not remove {d}/: {e}")
    
    # Run conversions for all datasets
    print("\n[*] Starting dataset conversions...\n")
    
    # Moving MNIST: Try NPY-based conversion first, fall back to sequence-based if needed
    npy_path = os.path.join(raw_data_root, "Moving_MNIST", "mnist_test_seq.npy")
    if os.path.exists(npy_path):
        print("[*] Found mnist_test_seq.npy - attempting direct NPY converter")
        success = convert_moving_mnist_from_npy(npy_path, os.path.join(dest_data_root, "moving_mnist"), max_sequences=None)
        if not success:
            print("[*] NPY converter failed - trying sequence-based converter")
            convert_moving_mnist(os.path.join(raw_data_root, "Moving_MNIST"), os.path.join(dest_data_root, "moving_mnist"))
    else:
        print("[*] NPY file not found - trying sequence-based converter")
        convert_moving_mnist(os.path.join(raw_data_root, "Moving_MNIST"), os.path.join(dest_data_root, "moving_mnist"))
    
    convert_dstext_v2(os.path.join(raw_data_root, "DSText_V2"), os.path.join(dest_data_root, "dstext_v2"))
    # convert_roadtext(os.path.join(raw_data_root, "RoadText"), os.path.join(dest_data_root, "roadtext"))
    convert_bovtext_or_icdar(os.path.join(raw_data_root, "BOVText"), os.path.join(dest_data_root, "bovtext"), "bovtext")
    convert_bovtext_or_icdar(os.path.join(raw_data_root, "ICDAR_SVT"), os.path.join(dest_data_root, "icdar_svt"), "icdar_svt")
    
    # Print summary and exit with appropriate code
    overall_success = print_conversion_summary()
    
    if overall_success:
        print("\n[✓] Video dataset conversion completed with some successes!")
        return 0
    else:
        print("\n[!] Video dataset conversion failed - no datasets were successfully converted.")
        print("[*] Please check:")
        print("    1. That data/ directories contain the raw video data")
        print("    2. That annotation files exist (XML for Moving MNIST/DSText V2, JSON for RoadText/BOVText/ICDAR)")
        print("    3. Video file formats are compatible with OpenCV")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

