import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data.physionet_loader import load_physionet
from models.classifier import ECGClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 1248
BATCH_SIZE = 32
EPOCHS = 15

REAL_PATH = "Rawdata/Physionet"
FAKE_PATH = "generated_data/generated_ecg_2000.npy"


# ----------------------
# Load Real ECG
# ----------------------
print("Loading real ECG...")
X_real = load_physionet(
    REAL_PATH,
    seq_len=SEQ_LEN,
    lowcut=0.5,
    highcut=40.0,
    lead_index=0
)

X_real = X_real[:, :, 0]
y_real = np.zeros(len(X_real))


# ----------------------
# Load Fake ECG
# ----------------------
print("Loading fake ECG...")
X_fake = np.load(FAKE_PATH)
X_fake = X_fake[:, 0, :]
y_fake = np.ones(len(X_fake))


# ----------------------
# Balance Dataset
# ----------------------
min_samples = min(len(X_real), len(X_fake))
X_real = X_real[:min_samples]
X_fake = X_fake[:min_samples]
y_real = y_real[:min_samples]
y_fake = y_fake[:min_samples]

X = np.concatenate([X_real, X_fake])
y = np.concatenate([y_real, y_fake])


# ----------------------
# Train/Test Split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_train = torch.tensor(X_train).unsqueeze(1).float()
X_test = torch.tensor(X_test).unsqueeze(1).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE
)


# ----------------------
# Model
# ----------------------
model = ECGClassifier(seq_len=SEQ_LEN).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------
# Training
# ----------------------
print("Training classifier...")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")


# ----------------------
# Evaluation
# ----------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))