import pandas as pd
from configs import paths
import os
import numpy as np

def get_global_part_info(partID):
    """
    reads the subfiles of a part and returns a dataframe with the local position of the subfiles. 
    """
    path = os.path.join(paths.CONFIG["paths"]["ldraw"],"parts", f"{partID}.dat")
    subfiles = []

    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('1 '): 
                    subfiles.append(line.strip().split())
    except FileNotFoundError:
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Einlesen der Datei: {e}")
    subfiles_df = pd.DataFrame(subfiles, columns= ["type","color","x","y","z","a","b","c","d","e","f","g","h","i", "subfile"])
    return subfiles_df
    

def match_global_position(subfiles_df, model_df_line):
    """
    matches the local part position to the global position by applying the rotation and translation from the model_df_line.
    """
    R = np.array([[model_df_line["a"], model_df_line["b"], model_df_line["c"]],
                  [model_df_line["d"], model_df_line["e"], model_df_line["f"]],
                  [model_df_line["g"], model_df_line["h"], model_df_line["i"]]])
    t = np.array([model_df_line["x"], model_df_line["y"], model_df_line["z"]])


    pos_local = subfiles_df[["x", "y", "z"]].values.astype(float)
    pos_world = (R @ pos_local.T).T + t
    subfiles_df[["x_world", "y_world", "z_world"]] = pos_world

    return subfiles_df


if __name__ == "__main__":

    model_path = paths.CONFIG["paths"]["ready"]
    model_df = pd.read_csv(os.path.join(model_path, "20006-1.csv"))

    partID = model_df["part"].iloc[25]
    subfiles_df = get_global_part_info(partID)
    print(f"PartID:{partID}")

    subfiles_df = get_global_part_info(partID)
    print(subfiles_df.head())
    print(model_df.iloc[25])
    df_global = match_global_position(subfiles_df, model_df.iloc[25])
    print(df_global.head())

