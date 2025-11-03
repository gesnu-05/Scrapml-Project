♻️ ML Scrap Classification Project
🧩 Overview

This project automatically classifies waste items (scrap materials) into seven categories using deep learning and computer vision. It mimics an industrial conveyor belt that classifies incoming items in real time, logging results and confidence levels.

🧠 Dataset Used
📘 Dataset Description

The dataset contains ~37,844 labeled images divided into 7 categories:

Battery

Biological

Clothes

Metal

Plastic

Shoes

Trash

Each image shows a single object on a neutral background (e.g., cans, shoes, clothes, batteries, etc.).

🎯 Why This Dataset?

Represents real-world recyclable and non-recyclable materials.

Ideal for simulating an automated waste segregation system.

Balanced and suitable for industrial ML applications.

🧱 Architecture & Training Process
🧠 Model Architecture

Base Model: ResNet-18 (pretrained on ImageNet)

Approach: Transfer Learning (only the final fully connected layer retrained for 7 classes)

Framework: PyTorch (CPU build)

Input Size: 224×224×3

⚙️ Training Pipeline
🧩 Data Augmentation

Random horizontal flips

Random rotations

Normalization
→ Enhances generalization to lighting and background variations.

🧪 Data Split
Split	Count	Percentage
Train	26,490	70%
Validation	5,676	15%
Test	5,678	15%
⚙️ Configuration
Parameter	Value
Optimizer	Adam
Learning Rate	1e-4
Loss Function	CrossEntropyLoss
Early Stopping	Enabled (based on validation accuracy)
Checkpoint	best_model.pt (best validation accuracy)
💾 Output Models

best_model.pt → Main trained model

model_torchscript.pt → TorchScript portable version

🧮 Model Summary
Layer Type	Details
Convolutional Layers	Extract low/mid-level features
Residual Blocks	Improve gradient flow & convergence
Fully Connected (FC)	512 → 7 neurons
Activation	ReLU
Output	Softmax (7 classes)
🧩 Deployment Decisions
Component	Decision	Reason
Format	TorchScript (.pt)	Portable, optimized for CPU
Inference Engine	PyTorch runtime	Compatible with VS Code
Simulation	Python loop (frame-wise)	No camera dependency
Confidence Threshold	0.85	Reduces false positives
Active Learning	Auto-saves low-confidence or wrong predictions	Enables incremental retraining
🗂️ Folder Structure
ML_Scrap_Classification/
│
├── data/
│   ├── raw/
│   │   ├── Battery/
│   │   ├── Biological/
│   │   ├── Clothes/
│   │   ├── Metal/
│   │   ├── Plastic/
│   │   ├── Shoes/
│   │   └── Trash/
│   ├── processed/
│   └── retrain/
│       └── misclassified/
│
├── models/
│   ├── best_model.pt
│   ├── model_torchscript.pt
│   └── fine_tuned_model.pt
│
├── results/
│   ├── confusion_matrix.png
│   └── conveyor_results.csv
│
├── src/
│   ├── dataset_preparation.py
│   ├── train_model.py
│   ├── inference_robust.py
│   ├── retrain_model.py
│   └── conveyor_simulation.py
│
└── README.md

⚙️ How to Run
1️⃣ Setup
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib tqdm pillow scikit-learn

2️⃣ Dataset Preparation

Place all folders inside data/raw/ and run:

python src/dataset_preparation.py


Output:

✅ Dataset prepared!
Train: 26490 images
Val:   5676 images
Test:  5678 images
Classes: ['Battery', 'Biological', 'Clothes', 'Metal', 'Plastic', 'Shoes', 'Trash']

3️⃣ Model Training
python src/train_model.py


Output:
✅ Model saved to models/best_model.pt and model_torchscript.pt
Accuracy: ~92%

4️⃣ Inference (Single Image)
python src/inference_robust.py --image data/raw/Plastic/plastic1.jpg --threshold 0.85 --save-uncertain


Sample Output:

📸 Image: data/raw/Plastic/plastic1.jpg
🔍 Predicted: Plastic
📊 Confidence: 0.9723
✅ Correct prediction: True
⚠️ Low confidence flag: False

5️⃣ Conveyor Simulation (CSV Logging)

Simulates conveyor belt scanning:

python src/conveyor_simulation.py --folder data/raw/Plastic --interval 1.0


Sample Output:

🚀 Starting conveyor simulation...
📸 plastic1.jpg -> Plastic (0.987)
📸 plastic2.jpg -> Metal (0.61) ⚠️
✅ Simulation complete. Results saved to results/conveyor_results.csv


CSV Example:

Frame	Predicted	Confidence	LowConfidence	TrueLabel	Timestamp
plastic1.jpg	Plastic	0.987	False	Plastic	2025-11-03 14:21:02
plastic2.jpg	Metal	0.610	True	Plastic	2025-11-03 14:21:03
📊 Performance Summary
Metric	Score
Accuracy	92%
Precision	0.92
Recall	0.91
F1-Score	0.92
Classes	7
Model	ResNet-18
🧩 Key Features

✅ Transfer Learning (ResNet-18)
✅ Early Stopping & Checkpoints
✅ TorchScript Deployment
✅ Confidence Thresholding
✅ Real-time Conveyor Simulation
✅ Active Learning & Retraining
✅ CSV Logging

🧾 Conclusion

This project demonstrates an end-to-end ML pipeline for automated scrap classification, covering:

Data preprocessing

Model training

Real-time inference

Active learning & retraining

Deployment-ready TorchScript export

With ~92% accuracy, this model provides a strong baseline for AI-powered waste segregation, scalable to real conveyor systems using Raspberry Pi and camera modules.

👨‍💻 Author

GESNU DHARRSHAN A (CSE)
ML Intern Assignment — 2025
Department of Computer Science and Engineering
