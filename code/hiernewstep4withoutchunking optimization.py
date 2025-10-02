import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

import gc
import psutil
import argparse
import joblib
import random
import numpy as np
import torch

# === Fixed seeds for reproducibility ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV with scaled features")
parser.add_argument("--model_out", required=True, help="Path to save model")
parser.add_argument("--encoder_out", required=True, help="Path to save label encoder")
parser.add_argument("--loss_out", required=True, help="CSV file to save epoch losses")
args = parser.parse_args()


class VirusMultiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def compute_hierarchical_metrics(true_labels, pred_labels):
    def get_prefix(label):
        return label.split('_')[0] if '_' in label else label

    true_prefixes = [get_prefix(lbl) for lbl in true_labels]
    pred_prefixes = [get_prefix(lbl) for lbl in pred_labels]

    correct = sum(1 for t, p in zip(true_prefixes, pred_prefixes) if t == p)
    total_pred = len(pred_prefixes)
    total_true = len(true_prefixes)

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


if __name__ == "__main__":
    print("🔄 Loading scaled k-mer dataset...")
    df = pd.read_csv(args.input).dropna()
    df["Label"] = df["Label"].astype(str)
    features = df.drop(columns=["Label"]).astype(np.float32).values
    labels = df["Label"].values
    input_dim = features.shape[1]
    del df
    gc.collect()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels).astype(np.int64)
    num_classes = len(label_encoder.classes_)
    joblib.dump(label_encoder, args.encoder_out)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = DataLoader(VirusMultiDataset(X_train, y_train), batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(VirusMultiDataset(X_test, y_test), batch_size=32, pin_memory=True)

    # MI-based importance (optional regularization)
    X_mi, _, y_mi, _ = train_test_split(features, y, train_size=0.2, stratify=y, random_state=1)
    mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=False)
    mi_scores = mi_scores / np.sum(mi_scores)
    np.save("kmer_importance.npy", mi_scores)
    del X_mi, y_mi
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiClassClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    importance_vector = torch.tensor(mi_scores, dtype=torch.float32).to(device)
    lambda_reg = 0.1

    print("🚀 Starting training...")
    epoch_losses = []

    epoch_losses = []  # Initialize loss storage before loop
    for epoch in range(1, 101):
        # === Training phase ===
        model.train()
        train_loss = 0
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
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # === Validation phase ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, attn_weights = model(inputs)

                loss_ce = criterion(outputs, targets)
                batch_importance = importance_vector.unsqueeze(0).expand_as(attn_weights)
                loss_attn = nn.functional.mse_loss(attn_weights, batch_importance)
                loss = loss_ce + lambda_reg * loss_attn
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        epoch_losses.append({
            "Epoch": epoch,
            "TrainingLoss": avg_train_loss,
            "ValidationLoss": avg_val_loss
        })
        if epoch % 2 == 0:
            print(f"📉 Epoch {epoch}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

        # Evaluate on test set
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
        conf = confusion_matrix(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')

        all_targets_str = [label_encoder.inverse_transform([t])[0] for t in all_targets]
        all_preds_str = [label_encoder.inverse_transform([p])[0] for p in all_preds]
        h_prec, h_rec, h_f1 = compute_hierarchical_metrics(all_targets_str, all_preds_str)

        print(f"🧪 Evaluation at Epoch {epoch}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Hierarchical Precision: {h_prec:.4f}")
        print(f"Hierarchical Recall: {h_rec:.4f}")
        print(f"Hierarchical F1 Score: {h_f1:.4f}")
        print("Confusion Matrix:")
        print(conf)

        print(f"📉 Epoch {epoch}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")
        epoch_losses.append({"Epoch": epoch, "TrainingLoss": avg_train_loss, "ValidationLoss": avg_val_loss})

    # Save epoch loss CSV
    loss_df = pd.DataFrame(epoch_losses)
    loss_df.to_csv(args.loss_out, index=False)
    print(f"📈 Saved epoch losses to: {args.loss_out}")

    print("\n📊 Evaluating model...")
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
    conf = confusion_matrix(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')

    all_targets_str = [label_encoder.inverse_transform([t])[0] for t in all_targets]
    all_preds_str = [label_encoder.inverse_transform([p])[0] for p in all_preds]
    h_prec, h_rec, h_f1 = compute_hierarchical_metrics(all_targets_str, all_preds_str)

    print("\n✅ Multiclass Classification Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf)

    print(f"\nHierarchical Precision: {h_prec:.4f}")
    print(f"Hierarchical Recall: {h_rec:.4f}")
    print(f"Hierarchical F1 Score: {h_f1:.4f}")

    print("\n🧾 Label Encoding Mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i} = {label}")

    print(f"\n🧠 Peak Memory Used: {psutil.virtual_memory().used / 1e9:.2f} GB")

    # === Save Attention Weights vs. Mutual Information ===
    print("\n💾 Saving Attention Weights vs. Mutual Information...")

    # Compute average attention weights across all training batches
    model.eval()
    all_attn_weights = []

    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            _, attn_weights = model(inputs)
            all_attn_weights.append(attn_weights.cpu().numpy())

    # Stack and average
    avg_attn_weights = np.mean(np.vstack(all_attn_weights), axis=0)

    # Extract k-mer names (feature columns)
    kmer_names = pd.read_csv(args.input, nrows=1).drop(columns=["Label"]).columns

    # Save DataFrame
    attn_df = pd.DataFrame({
        "kmer": kmer_names,
        "MI_score": mi_scores,
        "Attention_weight": avg_attn_weights
    })
attn_df.to_csv("attention_vs_mi.csv", index=False)
print("✅ Saved 'attention_vs_mi.csv' with k-mer attention and MI scores.")

torch.save(model.state_dict(), args.model_out)
