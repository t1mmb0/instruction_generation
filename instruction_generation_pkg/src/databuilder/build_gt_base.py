from configs import paths
import os
import pandas as pd
import itertools

# -----------------------------
# Parameter
# -----------------------------

file_name = "75895-1"

# -----------------------------
# Load Ground Truth DataFrame Base
# -----------------------------

path = paths.CONFIG["paths"]["ready"]
gt_path = paths.CONFIG["paths"]["gt"]
df = pd.read_csv(path + "/" + file_name + ".csv")
pairs = []
for (i1, row1), (i2, row2) in itertools.combinations(df.iterrows(), 2):
    pairs.append({
            "part_id_1": row1["part_id"],
            "part_id_2": row2["part_id"],
            "part_num_1": row1["part"],
            "part_num_2": row2["part"],
            "connected": 0
        })
gt_base = pd.DataFrame(pairs)
gt_base.to_csv(gt_path + "/" + "empty_gt_" + file_name + ".csv", index=False)

print(f"Ground Truth Base for {file_name} created!")