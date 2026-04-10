import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_digit_model() -> nn.Module:
    # Use an internet-downloaded pretrained ImageNet backbone (ResNet18) and adapt to 10 digits
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    return model


def load_classifier(model_path: str, data_dir: str, epochs: int = 3, batch_size: int = 64) -> nn.Module:
    model_path = str(model_path)
    model = build_digit_model()
    device = get_device()
    model = model.to(device)

    if os.path.exists(model_path):
        print(f"Loading existing digit classifier weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    print("No existing digit classifier found, training a new one from classification data.")

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Could not find classification data folder: {data_dir}")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    total = len(dataset)
    if total == 0:
        raise RuntimeError(f"No labeled digits found in {data_dir}. Run data generation first.")

    # quick split 90/10
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    model.train()
    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item() * images.size(0))
            preds = outputs.argmax(dim=1)
            correct += int((preds == labels).sum())
            count += images.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        acc = correct / count if count > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += int((preds == labels).sum())
                val_count += images.size(0)

        val_acc = val_correct / val_count if val_count > 0 else 0.0
        print(f"Epoch {ep+1}/{epochs}: loss={avg_loss:.4f}, train_acc={acc:.4f}, val_acc={val_acc:.4f}")
        model.train()

    torch.save(model.state_dict(), model_path)
    print(f"Saved trained digit classifier to: {model_path}")

    model.eval()
    return model


def preprocess_crop(img: np.ndarray, bbox: tuple) -> torch.Tensor:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]

    x1 = int(max(0, np.floor(x1)))
    y1 = int(max(0, np.floor(y1)))
    x2 = int(min(w, np.ceil(x2)))
    y2 = int(min(h, np.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox {bbox} for image shape {img.shape}")

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Crop has zero size")

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_crop = T.ToPILImage()(crop)

    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(pil_crop)


def predict_on_image(model: nn.Module, image_path: str, yolo_df: pd.DataFrame, output_csv: str):
    device = get_device()
    model = model.to(device)
    model.eval()

    rows = []
    grouped = yolo_df.groupby("image_path")
    for image_path, group in grouped:
        if not os.path.exists(image_path):
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue

        for _, row in group.iterrows():
            if pd.isna(row['pred_x1']) or pd.isna(row['pred_y1']):
                continue
            bbox = (row['pred_x1'], row['pred_y1'], row['pred_x2'], row['pred_y2'])
            try:
                inputs = preprocess_crop(img, bbox).unsqueeze(0).to(device)
            except Exception as ex:
                print(f"Skipping crop for row {row.name}: {ex}")
                continue

            with torch.no_grad():
                out = model(inputs)
                probs = torch.softmax(out, dim=-1)
                pred = int(probs.argmax(dim=-1).item())
                conf = float(probs.max().item())

            rows.append({
                "sample_id": row['sample_id'],
                "category": row['category'],
                "image_path": row['image_path'],
                "pred_x1": row['pred_x1'],
                "pred_y1": row['pred_y1'],
                "pred_x2": row['pred_x2'],
                "pred_y2": row['pred_y2'],
                "pred_conf": row['pred_conf'],
                "digit": pred,
                "digit_conf": conf,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Digit predictions saved to: {output_csv}")
    return out_df


def create_labeled_images(predictions_df: pd.DataFrame, output_dir: str):
    out_visual = os.path.join(output_dir, "digit_labeled")
    os.makedirs(out_visual, exist_ok=True)

    for image_path, group in predictions_df.groupby("image_path"):
        if not os.path.exists(image_path):
            continue
        img = cv2.imread(image_path)
        if img is None:
            continue

        for _, row in group.iterrows():
            if pd.isna(row['pred_x1']) or pd.isna(row['pred_y1']):
                continue
            x1, y1, x2, y2 = map(int, [row['pred_x1'], row['pred_y1'], row['pred_x2'], row['pred_y2']])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            label = f"{int(row['digit'])} ({row['digit_conf']:.2f})"
            cv2.putText(img, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out_path = os.path.join(out_visual, os.path.basename(image_path))
        cv2.imwrite(out_path, img)

    print(f"Annotated digit-labeled images saved to: {out_visual}")


def main():
    parser = argparse.ArgumentParser(description="Digit recognition for YOLO detections")
    parser.add_argument("--yolo-csv", default=os.path.join("outputs", "bbox_comparison", "yolo_predictions.csv"), help="YOLO predictions CSV path")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "bbox_comparison"), help="Output directory")
    parser.add_argument("--model-path", default=os.path.join("outputs", "bbox_comparison", "digit_classifier.pth"), help="Saved digit classifier weights")
    parser.add_argument("--classification-data", default=os.path.join("data", "classification", "single_digits"), help="Digit label training data")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs to train digit classifier when missing")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    yolo_csv = args.yolo_csv
    output_dir = args.output_dir

    if not os.path.exists(yolo_csv):
        raise FileNotFoundError(f"YOLO predictions file not found: {yolo_csv}")

    yolo_df = pd.read_csv(yolo_csv)
    model = load_classifier(args.model_path, args.classification_data, epochs=args.epochs, batch_size=args.batch_size)

    digit_csv = os.path.join(output_dir, "digit_predictions.csv")
    predictions_df = predict_on_image(model, yolo_csv, yolo_df, digit_csv)
    create_labeled_images(predictions_df, output_dir)


if __name__ == "__main__":
    main()
