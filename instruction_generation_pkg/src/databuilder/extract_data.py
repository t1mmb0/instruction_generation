from configs import paths
import os
import json
import pandas as pd



root_dir = paths.CONFIG["paths"]["raw"]

def get_model_dirs(root_dir):
    model_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            model_dirs.append(os.path.join(root, name))
    
    return model_dirs

def extract_all(root_dir):
    models = []
    valid_models = []
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            models.append(os.path.join(root, name))

    for model in models:
        assembly_file = extract_assembly_file(model)
        data = read_assembly_file(assembly_file)
        if is_valid_assembly(data):
            valid_models.append(model)
    print(f"Extracted {len(models)} models, {len(valid_models)} are valid.")
    return valid_models

def extract_one(model_dir):
        assembly_file = extract_assembly_file(model_dir)
        data = read_assembly_file(assembly_file)
        return data


def extract_assembly_file(model_dir):
    assembly_file = os.path.join(model_dir, "assembly.json")
    if not os.path.exists(assembly_file):
        raise FileNotFoundError(f"Assembly file not found: {assembly_file}")
    
    return assembly_file

def read_assembly_file(assembly_file):
    with open(assembly_file, "r") as f:
        data = json.load(f)
        
    return data

def is_valid_assembly(data):
    joints = data.get("joints") or {}
    return len(data["occurrences"]) > 5 and len(joints) > 4

def extract_occurrences(data):
    rows = []
    for uuid, occ in data["occurrences"].items():
        transform = occ["transform"]
        props = occ["physical_properties"]
        rows.append({
            "part_id": uuid,
            "name": occ["name"],
            "x": transform["origin"]["x"],
            "y": transform["origin"]["y"],
            "z": transform["origin"]["z"],
            "a": transform["x_axis"]["x"],
            "b": transform["x_axis"]["y"],
            "c": transform["x_axis"]["z"],
            "d": transform["y_axis"]["x"],
            "e": transform["y_axis"]["y"],
            "f": transform["y_axis"]["z"],
            "g": transform["z_axis"]["x"],
            "h": transform["z_axis"]["y"],
            "i": transform["z_axis"]["z"],
            "part": occ["component"],
            "mass": props["mass"],
            "volume": props["volume"],
            "area": props["area"],
        })
    return pd.DataFrame(rows).reset_index(drop=True)
if __name__ == "__main__":

    valid_models = extract_all(root_dir)

    model_dir = valid_models[0]
    print(f"Processing model directory: {model_dir}")
    data = extract_one(model_dir)

    occurrences_df = extract_occurrences(data)
    occurrences_df.to_csv(os.path.join(paths.CONFIG["paths"]["ready"], "occurrences.csv"), index=False)
    print("Joints:", len(data.get("joints") or {}))
    print("Contacts:", data.get("contacts"))
    print("Holes:", len(data.get("holes") or []))
    print(data["holes"][0])