import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import gc
import psutil
import argparse
import os
import pickle

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV with scaled k-mer features")
parser.add_argument("--model_out", required=True, help="Path to save model weights (.pt)")
parser.add_argument("--encoder_out", required=True, help="Path to save label encoder (.pkl)")
args = parser.parse_args()

# === Derive .npy filename from model_out name ===
base = os.path.splitext(os.path.basename(args.model_out))[0]
npy_filename = f"kmer_importance_{base}.npy"

# === I. Dataset Class ===
class VirusMultiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === II. Neural Network Class ===
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        attn_weights = torch.sigmoid(self.attn(x))
        x_weighted = x * attn_weights
        logits = self.classifier(x_weighted)
        return logits, attn_weights

label_to_path = {
    # Level 1
    'Virus': ['Virus'],
    'ERV': ['ERV'],
    'NONERV': ['NONERV'],

    # Level 2 - Baltimore Classes
    'C1': ['Virus', 'C1'],
    'C2': ['Virus', 'C2'],
    'C3': ['Virus', 'C3'],
    'C4': ['Virus', 'C4'],
    'C5': ['Virus', 'C5'],
    'C6': ['Virus', 'C6'],
    'C7': ['Virus', 'C7'],

    # Level 3 - Species (under each BC)
    'African_swine_fever_virus': ['Virus', 'C1', 'African_swine_fever_virus'],
    'Human_papillomaviruses(HPV)': ['Virus', 'C1', 'Human_papillomaviruses(HPV)'],
    'HumanHeprevirus': ['Virus', 'C1', 'HumanHeprevirus'],

    'Torque_teno_virus': ['Virus', 'C2', 'Torque_teno_virus'],
    'Protoparvovirus': ['Virus', 'C2', 'Protoparvovirus'],
    'Canine_parvovirus': ['Virus', 'C2', 'Canine_parvovirus'],

    'Bluetongue_Virus': ['Virus', 'C3', 'Bluetongue_Virus'],
    'RotaVirus': ['Virus', 'C3', 'RotaVirus'],
    'Infectious_bursal_disease_virus': ['Virus', 'C3', 'Infectious_bursal_disease_virus'],

    'Dengue_Virus_Type_1': ['Virus', 'C4', 'Dengue_Virus_Type_1'],
    'Hepatitis_C_virus': ['Virus', 'C4', 'Hepatitis_C_virus'],
    'Norovirus': ['Virus', 'C4', 'Norovirus'],

    'HantaVirus': ['Virus', 'C5', 'HantaVirus'],
    'Influenza__A_Virus': ['Virus', 'C5', 'Influenza_A_Virus'],
    'Measles_Virus': ['Virus', 'C5', 'Measles_Virus'],

    'Bovine__diarrhea_virus': ['Virus', 'C6', 'Bovine_diarrhea_virus'],
    'HIV_I': ['Virus', 'C6', 'HIV_I'],
    'HTLV_I': ['Virus', 'C6', 'HTLV_I'],

    'Badnavirus': ['Virus', 'C7', 'Badnavirus'],
    'Cauliflower652+Dahli75a_mosaic_Virus': ['Virus', 'C7', 'Cauliflower652+Dahli75a_mosaic_Virus'],
    'Hypathitas_B_Virus': ['Virus', 'C7', 'Hypathitas_B_Virus'],
}

def hierarchical_metrics(y_true, y_pred, label_encoder):
    correct = 0
    total_precision = 0
    total_recall = 0
    N = len(y_true)

    for yt, yp in zip(y_true, y_pred):
        true_label = label_encoder.inverse_transform([yt])[0]
        pred_label = label_encoder.inverse_transform([yp])[0]

        true_path = set(label_to_path.get(true_label, []))
        pred_path = set(label_to_path.get(pred_label, []))

        intersection = len(true_path & pred_path)
        precision = intersection / len(pred_path) if pred_path else 0
        recall = intersection / len(true_path) if true_path else 0

        total_precision += precision
        total_recall += recall

    h_precision = total_precision / N
    h_recall = total_recall / N
    h_f1 = (2 * h_precision * h_recall) / (h_precision + h_recall + 1e-8)

    return h_precision, h_recall, h_f1

# === IV. Load Data ===
print("🔄 Loading scaled k-mer dataset...")
df = pd.read_csv(args.input).dropna()
df["Label"] = df["Label"].astype(str)
features = df.drop(columns=["Label"]).astype(np.float32).values
labels = df["Label"].values
X = features
input_dim = X.shape[1]
print(f"ℹ️ Input feature dimension: {input_dim}")
del df
gc.collect()

# === Label Encoding ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels).astype(np.int64)
num_classes = len(label_encoder.classes_)

with open(args.encoder_out, "wb") as f:
    pickle.dump(label_encoder, f)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_loader = DataLoader(VirusMultiDataset(X_train, y_train), batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(VirusMultiDataset(X_test, y_test), batch_size=32, pin_memory=True, num_workers=0)

# === Mutual Information-based Attention Supervision ===
print("🔍 Computing MI-based feature importance...")
X_mi, _, y_mi, _ = train_test_split(X, y, train_size=0.2, stratify=y, random_state=1)
mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=False)
mi_scores = mi_scores / np.sum(mi_scores)
np.save(npy_filename, mi_scores)
print(f"✅ MI scores saved and normalized to {npy_filename}.")
del X_mi, y_mi
gc.collect()

# === Model Initialization ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiClassClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
importance_vector = torch.tensor(mi_scores, dtype=torch.float32).to(device)
lambda_reg = 0.1

# === Training Loop ===
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, attn_weights = model(inputs)

        loss_ce = criterion(outputs, targets)
        batch_importance = importance_vector.unsqueeze(0).expand_as(attn_weights)
        loss_attn = nn.functional.mse_loss(attn_weights, batch_importance)
        loss = loss_ce + lambda_reg * loss_attn

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 2 == 0:
        print(f"\n📉 Epoch {epoch}: Total Loss = {total_loss:.2f}, CE = {loss_ce.item():.2f}, ATTN = {loss_attn.item():.2f}")

        # === Evaluation ===
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        h_precision, h_recall, h_f1 = hierarchical_metrics(all_targets, all_preds, label_encoder)

        print(f"🔁 Eval @ Epoch {epoch} — Accuracy: {acc:.4f}, F1: {f1:.4f}, HF1: {h_f1:.4f}")


# === Final Evaluation ===
print("\n📊 Final Evaluation...")
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

acc = accuracy_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds, average='weighted')
h_precision, h_recall, h_f1 = hierarchical_metrics(all_targets, all_preds, label_encoder)

print("\n✅ Final Results:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")
print(f"Hierarchical Precision: {h_precision:.4f}")
print(f"Hierarchical Recall: {h_recall:.4f}")
print(f"Hierarchical F1 (HF1): {h_f1:.4f}")

print("🧾 Label Encoding Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} = {label}")

torch.save(model.state_dict(), args.model_out)
print(f"✅ Model saved to {args.model_out}")
print(f"🧠 Peak Memory Used: {psutil.virtual_memory().used / 1e9:.2f} GB")
