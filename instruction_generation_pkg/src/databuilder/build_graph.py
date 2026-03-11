# ------------------------------------------------
# Step 0: IMPORTS
# ------------------------------------------------
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from configs.paths import CONFIG
import os

path_parts = CONFIG["paths"]["parts"]
path_joints = CONFIG["paths"]["joints"]

# ------------------------------------------------
# Step 1: DATA LOADER
# ------------------------------------------------

class GraphDataBuilder():
    def __init__(self, scaler, splitter, dataset_ids: tuple):
        self.splitter = splitter
        self.scaler = scaler
        self.dataset_ids = dataset_ids
        self.graphs = list()
        self.train = list()
        self.val = list()
        self.test = list()

    def _build_edge_index(self, dataset_id, mapping):
        edges_df = pd.read_csv(os.path.join(path_joints, dataset_id))
        pos_edge_pairs = edges_df[["part_id_1", "part_id_2"]].copy()
        pos_edge_pairs["part_id_1"] = pos_edge_pairs["part_id_1"].map(mapping)
        pos_edge_pairs["part_id_2"] = pos_edge_pairs["part_id_2"].map(mapping)
        edge_index_pos = torch.tensor(pos_edge_pairs.values.T, dtype=torch.long)
        edge_index_undir = torch.cat([edge_index_pos, edge_index_pos.flip(0)], dim=1)
        return edge_index_undir
    
    def _build_graph(self,dataset_id,):
        parts = pd.read_csv(os.path.join(path_parts, dataset_id)).sort_values("part_id").reset_index(drop=True)
        x = self.scaler.transform_to_tensor(dataset_id)
        mapping = part_id_mapping(parts)
        graph_data = Data(x=x, edge_index=self._build_edge_index(dataset_id, mapping))
        graph_data.model_id = dataset_id

        # Position (N, 3)
        graph_data.pos = torch.tensor(parts[["x","y","z"]].values, dtype=torch.float32)
        return graph_data

    def _build_all_graphs(self,):
        for dataset_id in self.dataset_ids:
            graph = self._build_graph(dataset_id)
            self.graphs.append(graph)
            
    def split_graph(self,graph):
        train, val, test = self.splitter(graph)
        return train, val, test

    def build_dataset(self,):
        self._build_all_graphs()
        for graph in self.graphs:
            train, val, test = self.split_graph(graph)
            self.train.append(train)
            self.val.append(val)
            self.test.append(test)
        return self
    
# ------------------------------------------------
# Step 2: HELPER FUNCTIONS
# ------------------------------------------------

def part_id_mapping(parts_df):
    part_ids = parts_df["part_id"].unique()
    mapping = {part_id: idx for idx, part_id in enumerate(part_ids)}
    return mapping

# ------------------------------------------------
# Step 3: FUNCTIONALITY TESTING
# ------------------------------------------------
from src.preprocessing.data_utils import GlobalScaler

if __name__ == "__main__":
    print("### FUNCTIONALITY TESTING ###")

    dataset_ids = []
    for root, dirs, files in os.walk(path_parts):
        for name in files:
            dataset_ids.append(os.path.join(name))

    splitter = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
    )

    global_scaler = GlobalScaler().fit(dataset_ids[0:2])
    builder = GraphDataBuilder(global_scaler, splitter, (dataset_ids[0], dataset_ids[1]))
    builder.build_dataset()

    print(builder.train)