# ------------------------------------------------
# Step 0: IMPORTS
# ------------------------------------------------
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from configs.paths import CONFIG
import os

path = CONFIG["paths"]["ready"]
gt_path = CONFIG["paths"]["gt"]

# ------------------------------------------------
# Step 2: DATA LOADER
# ------------------------------------------------

class GraphDataBuilder():
    def __init__(self, scaler, splitter, model_ids: tuple):
        self.splitter = splitter
        self.scaler = scaler
        self.model_ids = model_ids
        self.graphs = list()
        self.train = list()
        self.val = list()
        self.test = list()

    def _build_edge_index(self, model_id):
        edges_df = pd.read_csv(os.path.join(gt_path, f"gt_{model_id}.csv"))
        pos_edge_pairs = edges_df.query("connected == 1")[["part_id_1", "part_id_2"]]
        edge_index_pos = torch.tensor(pos_edge_pairs.values.T, dtype=torch.long)
        edge_index_undir = torch.cat([edge_index_pos, edge_index_pos.flip(0)], dim=1)
        return edge_index_undir
    
    def _build_graph(self,model_id,):
        x = self.scaler.transform_to_tensor(model_id)
        graph_data = Data(x=x, edge_index=self._build_edge_index(model_id))
        graph_data.model_id = model_id
        return graph_data
    
    def _build_all_graphs(self,):
        for model_id in self.model_ids:
            graph = self._build_graph(model_id)
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
# Step X: Functionality Test
# ------------------------------------------------
from src.preprocessing.data_utils import GlobalScaler

if __name__ == "__main__":
    print("### FUNCTIONALITY TESTING ###")


    global_scaler = GlobalScaler().fit(("20009-1", "20006-1"))

    splitter = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
    )

    builder = GraphDataBuilder(global_scaler, splitter, ("20009-1", "20006-1"))
    builder.build_dataset()

    print(builder.train)