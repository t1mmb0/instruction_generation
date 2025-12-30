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
from src.training.regularizer import Regularizer
from src.training.schedulers.base import SchedulerBuilder, SchedulerType
from src.preprocessing.data_utils import GraphDataBuilder, GlobalScaler
from src.utils.general_utils import set_seed
from src.utils.visualize_results import Visualizer
from experiments.experiment_runner import ExperimentRunner
# -----------------------------
# 1. Parameters
# -----------------------------

set_seed(42)

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
train_loader = DataLoader(builder.train, batch_size=2, shuffle=True,)
val_loader = DataLoader(builder.val, batch_size=2, shuffle=False,)
test_loader = DataLoader(builder.test, batch_size=2, shuffle=False,)

# -----------------------------
# 5. Build Model
# -----------------------------
print("\n[3] Building Model ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use first graph to determine feature dimensions
in_channels = builder.train[0].num_features

model = GCN(
    in_channels=in_channels,
    hidden_channels=128,
    out_channels=64,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4,)
criterion = torch.nn.BCEWithLogitsLoss()

print(model)
print("  -> Model built successfully.")


# -----------------------------
# 6. Training
# -----------------------------
#Parameter:
max_epochs = 200
seeds= [1,2,3,4]

print("\n[4] Training Model ...")
early_stopping = Regularizer(patience = 30)
LRSchedulerBuilder = SchedulerBuilder(scheduler_type= SchedulerType.COSINE, T_max = max_epochs, eta_min = 0.00001)

runner = ExperimentRunner(trainer=Trainer, 
                          seeds = seeds, 
                          max_epochs=max_epochs,
                          train_loader = train_loader, 
                          val_loader = val_loader, 
                          model=model, 
                          optimizer=optimizer, 
                          criterion=criterion, 
                          device=device, 
                          Regularizer=early_stopping, 
                          LR_Scheduler_Builder= LRSchedulerBuilder
                          )

results = runner.run()



# -----------------------------
# 7. Evaluation
# -----------------------------
seed = runner.seeds[0]
trainer = results[seed]["trainer"]

from src.training.analysis.feature_importance import FeatureImportanceAnalyzer

feature_names = ["part_id","color","x","y","z","a","b","c","d","e","f","g","h","i","part","part_cat_id","year_from","year_to","dim1","dim2","dim3"]

analyzer = FeatureImportanceAnalyzer(
    model=trainer.model,
    data_loader=val_loader,
    feature_names=feature_names,  
    device=device,
    criterion=criterion
)

df_imp = analyzer.compute_importance()
dead = analyzer.find_dead_features(df_imp, threshold=1e-4)

print(df_imp.head(10))
print("Dead features:")
print(dead)

