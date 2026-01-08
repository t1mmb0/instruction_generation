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
from src.models.model import GCN
from src.training.trainer import Trainer
from src.training.regularizer import Regularizer
from src.training.schedulers.base import SchedulerBuilder, SchedulerType
from src.preprocessing.data_utils import GraphDataBuilder, GlobalScaler
from src.utils.general_utils import set_seed
from src.utils.visualize_results import Visualizer
from src.runners  import run_config, experiment_runner, component_factory, single_run

# -----------------------------
# 1. Parameters
# -----------------------------
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 5. Setup Parameters
# -----------------------------
print("\n[3] Setup Experiment Runner...")

# Model Parameters
in_channels = builder.train[0].num_features
hidden_channels = 128
out_channels = 64

# Optimizer Parameters
lr = 0.005
weight_decay = 5e-4
#LR Parameter
Stype = SchedulerType.COSINE
eta_min = 0.00001

# Further Parameters
criterion = torch.nn.BCEWithLogitsLoss()
patience = 30
max_epochs = 200
seeds= [1,2,3,4]



# Builder
def model_builder():
    return GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    ).to(device)

def optimizer_builder(model):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

def regularizer_builder():
    return Regularizer(patience = patience)

def lrScheduler_builder():
    return SchedulerBuilder(scheduler_type=Stype, T_max = max_epochs, eta_min = eta_min)



# -----------------------------
# 6. Setup Runner
# -----------------------------

config = run_config.RunConfig(
    seeds=seeds,
    max_epochs=max_epochs,
    compute_feature_importance=False,
    feature_names=None,
)

factory = component_factory.ComponentFactory(
    model_builder = model_builder,
    optimizer_builder = optimizer_builder,
    regularizer_builder = regularizer_builder,
    lr_scheduler_builder=lrScheduler_builder,
)

executor = single_run.SingleRunExecutor(
    trainer_cls=Trainer,
    factory=factory,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    max_epochs = max_epochs,
    criterion=criterion,
)

# -----------------------------
# 7. Experiment
# -----------------------------


runner = experiment_runner.ExperimentRunner(
    executor = executor,
    config = config,
)

results = runner.run()
print("Finished!")