import sys
import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt

# Add the 'src' directory to the Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from src.image_preprocessing.enhancer_factory import get_enhancer

def show_before_after(image_path, method="unsharp_mask", output_path=None, **kwargs):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Read original image using cv2
    original_img = cv2.imread(image_path)
    if original_img is None:
         print(f"Error: Could not read image at {image_path}")
         return
         
    # Convert from BGR to RGB for matplotlib
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Get enhancer
    try:
        enhancer = get_enhancer(method, **kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Applying enhancement: {method}...")
    # Apply enhancement
    try:
        enhanced_img = enhancer.enhance(original_img)
        enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error applying enhancement: {e}")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(enhanced_img_rgb)
    axes[1].set_title(f"Enhanced Image ({method})")
    axes[1].axis('off')

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        default_out = f"visualization_{method}.png"
        plt.savefig(default_out)
        print(f"Saved visualization to {default_out}")
        
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize image enhancement (before & after)")
    parser.add_argument("--image", type=str, help="Path to the input image. If not provided, a random image will be sampled.", default=None)
    parser.add_argument("--method", type=str, default="unsharp_mask", help="Enhancement method (default: unsharp_mask, others: clahe, esrgan, none)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the visualization image")
    
    args = parser.parse_args()
    
    image_path = args.image
    if image_path is None:
        # Sample a random image from typical directories
        search_dirs = [os.path.join(project_root, "data"), os.path.join(project_root, "assets")]
        found_images = []
        for d in search_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            found_images.append(os.path.join(root, f))
        
        if not found_images:
            print("No images found in standard directories. Please provide an --image argument.")
            return
            
        image_path = random.choice(found_images)
        print(f"Randomly selected image: {image_path}")

    show_before_after(image_path, method=args.method, output_path=args.output)

if __name__ == "__main__":
    main()
