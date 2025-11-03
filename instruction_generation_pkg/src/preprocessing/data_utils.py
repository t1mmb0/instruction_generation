# ------------------------------------------------
# Step 0: IMPORTS
# ------------------------------------------------
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from configs.paths import CONFIG
import os
path = CONFIG["paths"]["ready"]
# ------------------------------------------------
# Step 1: FIT GLOBAL SCALER
# ------------------------------------------------

class GlobalScaler():
    def __init__(self):
        self.scaler = None
        self.ref_columns = None

    def fit(self, model_ids: tuple,):
        data = None
        for model_id in model_ids:
            parts = pd.read_csv(os.path.join(path, f"df_{model_id}.csv"))
            X = parts.select_dtypes(include=("float64", "int64")) # filter non-numerical
            if data is None:
                data = X
            else:
                data = pd.concat([data, X], ignore_index=True)

        scaler = StandardScaler()
        scaler.fit(data.fillna(0.0))
        self.scaler = scaler
        self.ref_columns = data.columns
        return self

    def transform(self, model_id,):
        if self.ref_columns is None:
            raise Exception(f"The Scaler has not been fitted yet.")
        
        data = prepare_data(model_id, self.ref_columns)
        data = self.scaler.transform(data.fillna(0.0))

        df = pd.DataFrame(data, columns=self.ref_columns)
        return df
    
    def transform_to_tensor(self,model_id,):
        if self.ref_columns is None:
            raise Exception(f"The Scaler has not been fitted yet.")
        
        data = prepare_data(model_id, self.ref_columns)
        data = self.scaler.transform(data.fillna(0.0))

        X = torch.tensor(data, dtype=torch.float32)
        return X
    
    def print_scaler(self,):
        if self.scaler is None:
            raise Exception(f"The Scaler has not been fitted yet.")
        print("Feature count:", len(self.scaler.mean_))
        print("Mittelwerte:", self.scaler.mean_)
        print("Standardabweichungen:", self.scaler.scale_)

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
        edges_df = pd.read_csv(os.path.join(path, f"gt_{model_id}.csv"))
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
# Helping Functions
# ------------------------------------------------
def prepare_data(model_id, ref_columns):
    data = pd.read_csv(os.path.join(path, f"df_{model_id}.csv"))
    data = data.select_dtypes(include=["float64","int64"])
    data = data.reindex(columns=ref_columns, fill_value=0.0)
    return data

# ------------------------------------------------
# Step X: Functionality Test
# ------------------------------------------------
if __name__ == "__main__":
    print("### FUNCTIONALITY TESTING ###")

    global_scaler = GlobalScaler().fit(("20009-1", "20006-1"))

    splitter = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
    )

    builder = GraphDataBuilder(global_scaler, splitter, ("20009-1", "20006-1"))
    builder.build_dataset()

    print(builder.train)