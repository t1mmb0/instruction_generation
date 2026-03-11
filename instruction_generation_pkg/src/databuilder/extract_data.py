# ------------------------------------------------
# Step 0: IMPORTS
# ------------------------------------------------

from configs import paths
import os
import json
import pandas as pd

# ------------------------------------------------
# Step 1: DATA EXTRACTION
# ------------------------------------------------

def build_datasets(root_dir, output_dir, part_threshold=5, joint_threshold=4):
    valid_models = extract_all(root_dir, part_threshold, joint_threshold)

    for model_dir in valid_models:
        try:
            build_dataset(model_dir, output_dir)
        except Exception as e:
            print(f"Error processing {model_dir}: {e}")

def build_dataset(dataset_dir, output_dir):

    dataset_name = os.path.basename(dataset_dir)
    data = extract_one(dataset_dir)
    occurrences_df = extract_occurrences(data)
    joints_df =extract_joints(data)

    occurrences_df.to_csv(os.path.join(output_dir, "parts", f"{dataset_name}.csv"), index=False)
    joints_df.to_csv(os.path.join(output_dir, "joints", f"{dataset_name}.csv"), index=False)

    print(f"Saved dataset for {dataset_name} with {len(occurrences_df)} parts and {len(joints_df)} joints to {output_dir}.")

    return occurrences_df, joints_df

def extract_all(root_dir, part_threshold, joint_threshold):
    models = []
    valid_models = []
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            models.append(os.path.join(root, name))

    for model in models:        
        try:
            assembly_file = extract_assembly_file(model)
        except Exception as e:
            print(f"Error processing {model}: {e}")
            continue   
        data = read_assembly_file(assembly_file)
        if is_valid_assembly(data, part_threshold, joint_threshold):
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

def is_valid_assembly(data, part_threshold, joint_threshold):
    joints = data.get("joints") or {}
    return len(data["occurrences"]) > part_threshold and len(joints) > joint_threshold

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

    return pd.DataFrame(rows)

def extract_joints(data):
    rows = []
    for uuid, joint in data["joints"].items():
        rows.append({
            "part_id_1": joint["occurrence_one"],
            "part_id_2": joint["occurrence_two"],
            "joint_type": joint["joint_motion"]["joint_type"],
        })

    return pd.DataFrame(rows)

# ------------------------------------------------
# Step 2: HELPER FUNCTIONS
# ------------------------------------------------

root_dir = paths.CONFIG["paths"]["raw"]

def get_model_dirs(root_dir):
    model_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            model_dirs.append(os.path.join(root, name))
    
    return model_dirs

# ------------------------------------------------
# Step 3: TEST FUNCTIONALITY
# ------------------------------------------------

if __name__ == "__main__":

    output_dir = paths.CONFIG["paths"]["output"]
    root_dir = paths.CONFIG["paths"]["raw"]

    build_datasets(root_dir, output_dir, part_threshold=30, joint_threshold=20)
    