import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch

# ----------------------------
# PATHS
# ----------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ----------------------------
# TRANSFORMS
# ----------------------------
# Training transforms: includes augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),             # resize all images to same shape
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation/test transforms: no augmentation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# LOAD DATASET
# ----------------------------
# torchvision.datasets.ImageFolder expects folders by class name
full_dataset = datasets.ImageFolder(root=RAW_DATA_DIR, transform=train_transform)

# Train / validation / test split
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply test_transform to val/test datasets
val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

# ----------------------------
# SAVE SAMPLE LOADERS
# ----------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print(f"✅ Dataset prepared!")
print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")
print(f"Test:  {len(test_dataset)} images")
print(f"Classes: {full_dataset.classes}")

# Optional: Save a few processed batches for sanity check
sample_batch = next(iter(train_loader))
torch.save(sample_batch, os.path.join(PROCESSED_DATA_DIR, "sample_batch.pt"))