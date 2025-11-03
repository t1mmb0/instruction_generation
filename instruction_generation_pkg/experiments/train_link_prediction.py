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
from src.models.model import GCN
from src.training.trainer import Trainer
from src.preprocessing.data_utils import GraphDataBuilder, GlobalScaler
from configs import paths
import os

# -----------------------------
# 1. Parameters
# -----------------------------
model_ids = ("20006-1", "20009-1")
plots_path = paths.CONFIG["paths"]["plots"]
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
trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=50)

trainer.evaluate_model(test_loader = test_loader)

# -----------------------------
# 7. Evaluation
# -----------------------------

import matplotlib.pyplot as plt

plt.plot(trainer.history["train"], label="Train Loss")
plt.plot(trainer.history["val"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plots_path, f"loss_curve.png"), dpi=300, bbox_inches="tight")
print("plot saved successfully!")