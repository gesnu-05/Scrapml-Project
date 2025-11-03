"""
Real-time (simulated) conveyor loop for waste classification.

Features:
 - Reads images from a folder (simulating conveyor frames).
 - For each frame: classifies using a TorchScript model and logs results to CSV.
 - Displays low-confidence warnings.
 - Optional manual override (press keys to confirm/correct predictions).

Usage examples:
 - Auto-simulate from test images:
     python src/realtime_simulation.py --use-test
 - Manual mode (press 'y' or 'n' for each frame):
     python src/realtime_simulation.py --use-test --manual
 - Custom source folder:
     python src/realtime_simulation.py --source data/simulation_frames --interval 0.5
"""

import argparse
import os
import time
import csv
import random
from datetime import datetime
import shutil

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def load_class_names(data_root="data/raw"):
    """Load sorted class names from dataset root."""
    return sorted([d for d in os.listdir(data_root)
                   if os.path.isdir(os.path.join(data_root, d))])


def get_transform():
    """Same preprocessing used during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_image_pil(img_pil, model, device, transform):
    """Predict single image (PIL) and return top index + confidence."""
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    return top_idx, top_prob, probs


# ---------------------------------------------------
# Main Simulation Loop
# ---------------------------------------------------

def main():
    print("🚀 Script started - initializing simulation...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/simulation_frames",
                        help="Folder containing frames (images) for simulation")
    parser.add_argument("--use-test", action="store_true",
                        help="Use random test images from data/raw instead of a specific folder")
    parser.add_argument("--model", default="models/model_torchscript.pt",
                        help="Path to TorchScript model")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between frames")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Confidence threshold for low-confidence flag")
    parser.add_argument("--manual", action="store_true",
                        help="Enable manual override mode (keypress required)")
    parser.add_argument("--results-csv", default="results/simulation_results.csv",
                        help="CSV file to store predictions")
    parser.add_argument("--retrain-dir", default="data/retrain/misclassified",
                        help="Folder to move misclassified images")
    args = parser.parse_args()

    # Ensure necessary folders
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    os.makedirs(args.retrain_dir, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    print(f"✅ Model loaded from: {args.model}")

    # Prepare transforms and class names
    class_names = load_class_names("data/raw")
    transform = get_transform()
    print(f"🧩 Classes detected: {class_names}")

    # Prepare image list
    if args.use_test:
        # Load all images from data/raw subfolders
        all_images = []
        for cls in class_names:
            cls_folder = os.path.join("data/raw", cls)
            for fn in os.listdir(cls_folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_images.append((os.path.join(cls_folder, fn), cls))
        random.shuffle(all_images)
        frame_list = all_images
        print(f"🧠 Loaded {len(frame_list)} test images for simulation.")
    else:
        # Load from --source folder
        if not os.path.isdir(args.source):
            raise FileNotFoundError(f"❌ Source folder '{args.source}' not found. Create it or use --use-test.")
        files = [f for f in os.listdir(args.source) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        frame_list = [(os.path.join(args.source, f), None) for f in files]
        print(f"📸 Loaded {len(frame_list)} frames from {args.source}")

    # CSV setup
    csv_exists = os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not csv_exists:
            writer.writerow(["timestamp", "frame_path", "true_label", "predicted_label",
                             "confidence", "low_conf_flag", "manual_flag"])

    print("\n▶️ Starting simulation... Press Ctrl+C to stop.\n")

    # Main loop
    try:
        for idx, (frame_path, true_label) in enumerate(frame_list, start=1):
            pil_img = Image.open(frame_path).convert("RGB")
            top_idx, top_prob, _ = predict_image_pil(pil_img, model, device, transform)
            pred_label = class_names[top_idx]
            low_flag = top_prob < args.threshold
            timestamp = datetime.now().isoformat(timespec="seconds")

            print(f"[{timestamp}] Frame {idx}/{len(frame_list)} → "
                  f"Pred: {pred_label} | Conf: {top_prob:.3f} | LowConf: {low_flag}")

            # Write to CSV
            with open(args.results_csv, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, frame_path, true_label or "", pred_label,
                                 f"{top_prob:.4f}", int(low_flag), 0])

            # Manual mode (optional)
            if args.manual:
                cv_img = cv2.cvtColor(np.array(pil_img.resize((640, 480))), cv2.COLOR_RGB2BGR)
                cv2.putText(cv_img, f"Pred: {pred_label} ({top_prob:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(cv_img, "Press: [y]=accept [n]=misclassified [q]=quit",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Simulation", cv_img)
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    print("🛑 Quit requested by user.")
                    break
                elif key == ord('y'):
                    print("✅ Marked as accepted.")
                elif key == ord('n'):
                    dest = os.path.join(args.retrain_dir, os.path.basename(frame_path))
                    shutil.copy(frame_path, dest)
                    print(f"⚠️ Marked misclassified → copied to {dest}")
                cv2.destroyAllWindows()
            else:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user.")
    finally:
        print("\n✅ Simulation finished.\n")


# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------

if __name__ == "__main__":
    main()
