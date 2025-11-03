"""
Simple, robust inference script for a single image using a TorchScript model.
Supports nested folder structures (e.g., data/raw/plastic/plastic1.jpg).

Usage:
    python src/inference_clean.py --image data/raw/Battery/battery_1.jpg
Or, if no image is given, it automatically uses the latest image in data/raw.
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time
from glob import glob

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
MODEL_PATH = "models/model_torchscript.pt"
DATA_ROOT = "data/raw"
THRESHOLD = 0.8  # confidence threshold for display

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def find_class_folder(image_path):
    """
    Automatically infer the true class name from path.
    Example: data/raw/plastic/plastic1.jpg -> 'plastic'
    """
    parts = image_path.replace("\\", "/").split("/")
    for p in reversed(parts):
        if p.lower() in ["cardboard", "glass", "metal", "paper", "plastic", "trash", "battery"]:
            return p.lower()
    return None


def load_class_names(data_root=DATA_ROOT):
    """Load class names alphabetically from the first folder level."""
    return sorted([d for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d))])


def get_transform():
    """Preprocessing identical to training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict(image_path, model, device, class_names, transform):
    """Run single-image prediction."""
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    pred_class = class_names[top_idx]
    return pred_class, top_prob, probs


def find_latest_image(root="data/raw"):
    """Find the most recently modified image inside data/raw."""
    exts = (".jpg", ".jpeg", ".png")
    all_imgs = [f for f in glob(os.path.join(root, "**", "*"), recursive=True)
                if f.lower().endswith(exts)]
    if not all_imgs:
        return None
    return max(all_imgs, key=os.path.getmtime)


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simple robust inference for waste classification")
    parser.add_argument("--image", help="Path to image for prediction (auto-selects latest if not given)")
    parser.add_argument("--model", default=MODEL_PATH, help="TorchScript model path")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Confidence threshold display")
    parser.add_argument("--smooth", type=int, default=1, help="Average multiple predictions for stability")
    args = parser.parse_args()

    # Auto-select latest image if not provided
    if args.image is None:
        args.image = find_latest_image(DATA_ROOT)
        if args.image is None:
            print("❌ No image found in data/raw/. Please provide one using --image.")
            return
        print(f"📸 Auto-selected latest image: {args.image}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    class_names = load_class_names(DATA_ROOT)
    transform = get_transform()

    # Multi-pass smoothing
    smoothed_probs = np.zeros(len(class_names))
    for i in range(args.smooth):
        pred_class, conf, probs = predict(args.image, model, device, class_names, transform)
        smoothed_probs += probs
        time.sleep(0.05)
    smoothed_probs /= args.smooth

    # Results
    top_idx = int(np.argmax(smoothed_probs))
    top_prob = float(smoothed_probs[top_idx])
    predicted_class = class_names[top_idx]
    low_conf_flag = top_prob < args.threshold
    true_label = find_class_folder(args.image)
    correct_flag = (predicted_class == true_label)

    print("\n📸 Image:", args.image)
    print(f"🔍 Predicted: {predicted_class}")
    print(f"📊 Confidence: {top_prob:.4f}")
    print(f"🧾 True label (from folder): {true_label}")
    print(f"✅ Correct prediction: {correct_flag}")
    print(f"⚠️ Low confidence: {low_conf_flag}")

    print("\n✅ Inference complete.")


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    main()
