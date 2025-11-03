"""
Simple inference script for one image using the TorchScript model.

Usage (manual):
    python src/inference_script.py --image data/raw/Battery/battery_1.jpg --model models/model_torchscript.pt --threshold 0.6

If no --image is provided, it will automatically pick the latest image inside data/raw.
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from glob import glob

# -------------------------------------------------------
# Helper: load class names (alphabetical order)
# -------------------------------------------------------
def load_class_names(data_root="data/raw"):
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    return classes

# -------------------------------------------------------
# Image transform (must match training)
# -------------------------------------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------
def predict(image_path, model, device, class_names, transform):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    return class_names[top_idx], top_prob, probs

# -------------------------------------------------------
# Auto-select the latest image if not specified
# -------------------------------------------------------
def find_latest_image(root="data/raw"):
    exts = (".jpg", ".jpeg", ".png")
    all_imgs = [f for f in glob(os.path.join(root, "**", "*"), recursive=True)
                if f.lower().endswith(exts)]
    if not all_imgs:
        return None
    return max(all_imgs, key=os.path.getmtime)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--model", default="models/model_torchscript.pt", help="Path to TorchScript model")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for low-confidence flag")
    parser.add_argument("--data-root", default="data/raw", help="Root folder with class subfolders")
    args = parser.parse_args()

    # Auto-pick latest image if not provided
    if args.image is None:
        args.image = find_latest_image(args.data_root)
        if args.image is None:
            print("❌ No image found in data/raw/. Please provide one using --image.")
            return
        print(f"📸 Auto-selected latest image: {args.image}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load TorchScript model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    # Class names + transform
    class_names = load_class_names(args.data_root)
    transform = get_transform()

    # Predict
    predicted_class, confidence, _ = predict(args.image, model, device, class_names, transform)
    low_flag = confidence < args.threshold

    print(f"\n🖼️ Image: {args.image}")
    print(f"🔍 Predicted: {predicted_class}")
    print(f"📊 Confidence: {confidence:.4f}")
    print(f"⚠️ Low confidence flag: {low_flag}")

if __name__ == "__main__":
    main()
