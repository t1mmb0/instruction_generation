# ================================================================
#  GNN FOR LINK PREDICTION
# ================================================================

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

from model import GCN, Trainer
from graph import Graph


# -----------------------------
# Load Datasets
# -----------------------------
print("\n[1] Loading datasets ...")
model_df = pd.read_csv("./results/df_20006-1.csv", na_values=["", " "])
gt = pd.read_csv("./results/gt_20006-1.csv", na_values=["", " "])
print(f"  -> model_df: {model_df.shape} | gt: {gt.shape}")


# -----------------------------
# Build Graph Data Object
# -----------------------------
print("\n[2] Building Data object ...")

# Build feature matrix
Feature_Matrix = model_df.select_dtypes(include=["number"]).drop(columns=["part_id"], errors="ignore")
Feature_Matrix = Feature_Matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Standardisation
scaler = StandardScaler()
Feature_Matrix = scaler.fit_transform(Feature_Matrix.values)

x = torch.tensor(Feature_Matrix, dtype=torch.float32)

# Positive edges
pos_edges = gt[gt["connected"] == 1][["part_id_1", "part_id_2"]]
pos_edge_index = torch.tensor(pos_edges.values.T, dtype=torch.long)

# Undirected edge index
edge_index = torch.cat([pos_edge_index, pos_edge_index.flip(0)], dim=1)

# Build Data object
data = Data(x=x, edge_index=edge_index) 
print(f"  -> Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")


# -----------------------------
# Split Data
# -----------------------------
print("\n[3] Splitting Data (Train/Val/Test) ...")

splitter = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=True,
)

train_data, val_data, test_data = splitter(data)
print("  -> Finished splitting.")

#------------------------------
#Show Graph
#------------------------------
G = Graph(data=data)
G.visualize_interactive()

# -----------------------------
# Build Model
# -----------------------------
print("\n[4] Building Model ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

model = GCN(
    in_channels=data.num_features,
    hidden_channels=16,
    out_channels=4,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

print(model)
print("  -> Model built successfully.")


# -----------------------------
# Training
# -----------------------------
print("\n[5] Training Model ...")
trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, patience=30)

best_val_loss = trainer.fit(train_data=train_data, val_data=val_data, max_epochs=300)
print(f"  -> Best Validation Loss: {best_val_loss:.4f}")


# -----------------------------
# Evaluation
# -----------------------------
print("\n[6] Evaluating Model on Test Set ...")

metrics = trainer.evaluate(test_data, k_list=[1, 3, 10, 50])

print(
    "\nEvaluation Results"
    "\n------------------"
    f"\nROC-AUC:           {metrics['roc_auc']:.4f}")