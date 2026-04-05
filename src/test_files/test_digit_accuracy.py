import os
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Import the digit model loading function
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DigitRecognizer'))
from digit_recognizer import build_digit_model, get_device

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Digit Recognition Accuracy: {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()