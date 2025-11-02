# ================================================================
#  GNN FOR LINK PREDICTION
# ================================================================

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

from model import GCN, Trainer
from graph import Graph
from data_utils import GraphDataBuilder, GlobalScaler

# -----------------------------
# 1. Parameters
# -----------------------------
model_ids = ("20006-1", "20009-1")

splitter = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
)

# -----------------------------
# 2. Build Global Scaler
# -----------------------------
print("\n[1] Fitting Global Scaler ...")
global_scaler = GlobalScaler().fit(model_ids=model_ids)

# -----------------------------
# 3. Build Graphs and Datasets
# -----------------------------
print("\n[2] Building Graph Datasets ...")
builder = GraphDataBuilder(scaler=global_scaler, splitter=splitter, model_ids=model_ids)
builder.build_dataset()

# Check dataset stats
print(f"  -> {len(builder.train)} training graphs")
print(f"  -> {len(builder.val)} validation graphs")
print(f"  -> {len(builder.test)} test graphs")

# -----------------------------
# 4. Create DataLoaders
# -----------------------------
train_loader = DataLoader(builder.train, batch_size=2, shuffle=True)
val_loader = DataLoader(builder.val, batch_size=2, shuffle=False)
test_loader = DataLoader(builder.test, batch_size=2, shuffle=False)

# -----------------------------
# 5. Build Model
# -----------------------------
print("\n[3] Building Model ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use first graph to determine feature dimensions
in_channels = builder.train[0].num_features

model = GCN(
    in_channels=in_channels,
    hidden_channels=16,
    out_channels=4,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

print(model)
print("  -> Model built successfully.")

# -----------------------------
# 6. Training
# -----------------------------
print("\n[4] Training Model ...")
trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, patience=30)

for i, (train_data, val_data) in enumerate(zip(builder.train, builder.val)):
    print(f"\n--- Training on model {builder.model_ids[i]} ---")
    train_data, val_data = train_data.to(device), val_data.to(device)
    best_val_loss = trainer.fit(train_data, val_data, max_epochs=300)
    print(f"  -> Best Validation Loss: {best_val_loss:.4f}")

# -----------------------------
# 7. Evaluation
# -----------------------------
print("\n[5] Evaluating Model on Test Set ...")

for i, test_data in enumerate(builder.test):
    print(f"\n--- Evaluating model {builder.model_ids[i]} ---")
    test_data = test_data.to(device)
    metrics = trainer.evaluate(test_data, k_list=[1, 3, 10, 50])

    print(
        "\nEvaluation Results"
        "\n------------------"
        f"\nROC-AUC:           {metrics['roc_auc']:.4f}"
    )

    confusion_matrix_metrics = trainer.analyze_predictions(test_data, threshold=0.6)
    print("\n[ Confusion Matrix Metrics ]")
    print("──────────────────────────────────────────────")
    print(f"  True Positives (TP): {confusion_matrix_metrics['TP']:>6}")
    print(f"  False Positives (FP): {confusion_matrix_metrics['FP']:>6}")
    print(f"  False Negatives (FN): {confusion_matrix_metrics['FN']:>6}")
    print(f"  True Negatives (TN): {confusion_matrix_metrics['TN']:>6}")
    print("──────────────────────────────────────────────")
    print(f"  Precision: {confusion_matrix_metrics['precision']*100:6.2f}%")
    print(f"  Recall:    {confusion_matrix_metrics['recall']*100:6.2f}%")
    print(f"  F1 Score:  {confusion_matrix_metrics['f1']*100:6.2f}%")
    print("──────────────────────────────────────────────")
