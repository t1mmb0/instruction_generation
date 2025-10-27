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
# Parameters
# -----------------------------
model_ids = ("20006-1", "20009-1")

# Preparation Global Scaler

scaler = StandardScaler()
global_features = pd.DataFrame()
feature_columns = None
for model_id in model_ids:
    parts_df = pd.read_csv(f"./results/df_{model_id}.csv", na_values=["", " "])
    X_df = (
        parts_df.select_dtypes(include=["number"])
        .drop(columns=["part_id"], errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if feature_columns is None:
        feature_columns = X_df.columns  # beim ersten Modell speichern
    else:
        # Fehlende Spalten ergänzen, damit alle Modelle dieselben Spalten haben
        for col in feature_columns:
            if col not in X_df.columns:
                X_df[col] = 0.0  # fehlende Spalten mit 0 auffüllen
        X_df = X_df[feature_columns]  # gleiche Spaltenreihenfolge erzwingen

    global_features = pd.concat([global_features, X_df])

scaler.fit(global_features)
del global_features


# -----------------------------
# Build Graphs per Model
# -----------------------------


train_graphs = list()
val_graphs = list()
test_graphs = list()

for model_id in model_ids:
    print(f"\n[1] Loading {model_id} ...")
    parts_df = pd.read_csv(f"./results/df_{model_id}.csv", na_values=["", " "])
    edges_df = pd.read_csv(f"./results/gt_{model_id}.csv", na_values=["", " "])

    X_df = (
        parts_df.select_dtypes(include=["number"])
        .drop(columns=["part_id"], errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


    # -----------------------------
    # Transform Data
    # -----------------------------

    # gleiche Spalten wie beim Fitten erzwingen
    for col in feature_columns:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_df = X_df[feature_columns]

    X_scaled = scaler.transform(X_df.values)


    X_scaled = scaler.transform(X_df.values)
    x = torch.tensor(X_scaled, dtype=torch.float32)

    pos_edge_pairs = edges_df.query("connected == 1")[["part_id_1", "part_id_2"]]
    edge_index_pos = torch.tensor(pos_edge_pairs.values.T, dtype=torch.long)
    edge_index_undir = torch.cat([edge_index_pos, edge_index_pos.flip(0)], dim=1)

    graph_data = Data(x=x, edge_index=edge_index_undir)
    graph_data.model_id = model_id


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

    train_data, val_data, test_data = splitter(graph_data)

    train_graphs.append(train_data)
    val_graphs.append(val_data)
    test_graphs.append(test_data)

    print("  -> Finished splitting.")

    # -----------------------------
    # Debug
    # -----------------------------
    print(f"Message Passing Edges: {train_data.edge_index.size(1)}")
    print(f"Positive Edges: {train_data.edge_label.sum()}")
    print(f"Negative Edges: {len(train_data.edge_label)- train_data.edge_label.sum()}")


#------------------------------
#Show Graph
#------------------------------
G = Graph(data=graph_data)
#G.visualize_interactive()

# -----------------------------
# Build Model
# -----------------------------
print("\n[4] Building Model ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

model = GCN(
    in_channels=graph_data.num_features,
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