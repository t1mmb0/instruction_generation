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
gt_path = CONFIG["paths"]["gt"]
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
            parts = pd.read_csv(os.path.join(path, f"{model_id}.csv"))
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
# Helping Functions
# ------------------------------------------------
def prepare_data(model_id, ref_columns):
    data = pd.read_csv(os.path.join(path, f"{model_id}.csv"))
    data = data.select_dtypes(include=["float64","int64"])
    data = data.reindex(columns=ref_columns, fill_value=0.0)
    return data