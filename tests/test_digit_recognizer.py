import os
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys

# Add src to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'digit_recognizer'))

from digit_recognizer import build_digit_model, get_device
from utils.metrics import print_metrics_report

def main():
    # Paths
    base_dir = root_dir
    digit_weights = os.path.join(base_dir, "outputs", "bbox_comparison", "digit_classifier.pth")
    data_dir = os.path.join(base_dir, "data", "classification", "single_digits")

    if not os.path.exists(digit_weights):
        print(f"Digit classifier weights not found: {digit_weights}")
        return

    if not os.path.isdir(data_dir):
        print(f"Single digits data not found: {data_dir}")
        return

    # Load model
    print("Loading digit classifier...")
    model = build_digit_model()
    device = get_device()
    model.load_state_dict(torch.load(digit_weights, map_location=device))
    model = model.to(device)
    model.eval()

    # Load dataset
    print("Loading single digits test data...")
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Test
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print_metrics_report(all_labels, all_preds, title="Digit Recognition Evaluation")

if __name__ == "__main__":
    main()
