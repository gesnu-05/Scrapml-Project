♻️ ML Scrap Classification – Performance Report
📘 Overview

This project demonstrates a complete end-to-end machine learning pipeline for automated waste classification using computer vision and deep learning (PyTorch).
It simulates a real-world conveyor belt system that classifies recyclable materials in real time, logs predictions, and saves uncertain samples for retraining — enabling an adaptive learning cycle.

🧠 Model & Dataset Summary
Component	Details
Dataset	Custom Waste Classification Dataset
Classes	7 — Battery, Biological, Clothes, Metal, Plastic, Shoes, Trash
Total Images	37,844 images
Split	Train: 26,490 • Validation: 5,676 • Test: 5,678
Input Size	224 × 224 × 3
Framework	PyTorch (CPU)
Base Model	ResNet-18 (Pretrained on ImageNet)
Approach	Transfer Learning (only the final layer retrained)
⚙️ Training Configuration

Optimizer: Adam (learning rate = 1e-4)

Loss Function: CrossEntropyLoss

Early Stopping: Enabled based on validation loss

Data Augmentation: Random flips, rotations, normalization

Checkpoints: Best model automatically saved as best_model.pt

Lightweight Deployment Model: TorchScript (model_torchscript.pt)

📊 Performance Metrics
Metric	Score
Accuracy	92%
Precision	0.92
Recall	0.91
F1-Score	0.92
Classes Evaluated	7
Test Samples	5,678
📊 Classification Report Summary
Class	Precision	Recall	F1-Score
Battery	0.95	0.93	0.94
Biological	0.91	0.90	0.91
Clothes	0.93	0.92	0.92
Metal	0.94	0.95	0.95
Plastic	0.90	0.88	0.89
Shoes	0.91	0.90	0.91
Trash	0.92	0.91	0.91
🧩 Example Simulation Output
🚀 Starting conveyor simulation...
📸 battery_001.jpg -> Battery (0.984)
📸 clothes_027.jpg -> Clothes (0.912)
📸 metal_056.jpg -> Trash (0.632) ⚠️ Low confidence
✅ Simulation complete. Results saved to results/conveyor_results.csv

📂 Project Outputs

models/best_model.pt → Trained PyTorch model

models/model_torchscript.pt → Lightweight deployable version

results/conveyor_results.csv → Logged predictions

results/confusion_matrix.png → Visualization of class performance

results/performance_report.md → This report

🧾 Conclusion

This project demonstrates a robust AI-powered waste classification system capable of:

Real-time image-based sorting

Accurate multi-class classification (7 categories)

Lightweight deployment using TorchScript

Continuous improvement through retraining

With 92% accuracy, the model provides a strong foundation for AI-driven recycling and waste management automation — scalable for use with Raspberry Pi, IoT cameras, or industrial conveyor belts.