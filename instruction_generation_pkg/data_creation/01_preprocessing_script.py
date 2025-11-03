import pandas as pd
import os
import re
import time
from configs.paths import CONFIG
import requests

raw_data_path = CONFIG["paths"]["raw_data"]
processed_data_path = CONFIG["paths"]["processsed"]

with open("API_KEY.txt", "r") as f:
    API_KEY = f.read().strip()


def parse_model_to_df(filepath):
    """
    Liest eine .ldr-Datei im LDraw-Format ein und gibt einen DataFrame mit allen Bauteilzeilen (Typ 1) zurück.

    Parameter:
    ----------
    filepath : str
        Pfad zur .ldr-Datei

    Rückgabe:
    ---------
    df : pd.DataFrame
        DataFrame mit Spalten: color, x, y, z, a-i (Matrix), part
    """
    part_lines = []

    # Datei einlesen
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('1 '):  # Nur Teilezeilen (Typ 1)
                    part_lines.append(line.strip())
    except FileNotFoundError:
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Fehler beim Einlesen der Datei: {e}")

    # Zerlegen in strukturierte Daten
    data = []
    for line in part_lines:
        parts = line.split()
        if len(parts) >= 15:
            data.append({
                'color': int(parts[1]),
                'x': float(parts[2]),
                'y': float(parts[3]),
                'z': float(parts[4]),
                'a': float(parts[5]), 'b': float(parts[6]), 'c': float(parts[7]),
                'd': float(parts[8]), 'e': float(parts[9]), 'f': float(parts[10]),
                'g': float(parts[11]), 'h': float(parts[12]), 'i': float(parts[13]),
                'part': parts[14]
            })

    # In DataFrame umwandeln
    df = pd.DataFrame(data)

    return df

def extract_dimensions(name):
    # 3D-Maße (z. B. 1 x 1 x 2/3)
    match3 = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*([\d/]+)", name)
    if match3:
        return match3.groups()

    # 2D-Maße (z. B. 2 x 4 oder 18 x 14)
    match2 = re.search(r"(\d+)\s*x\s*(\d+)", name)
    if match2:
        return (*match2.groups(), None)

    return (None, None, None)

def extract_bracket_info(name):
    bracket_match = re.search(r"\[(.*?)\]", name)
    bracket_text = bracket_match.group(1) if bracket_match else None

    return bracket_text


folder = "./data/raw"
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


for lego_model in files:

    model_dataframe = parse_model_to_df(os.path.join(raw_data_path, lego_model))
    model_dataframe["part"] = model_dataframe["part"].apply(lambda x: os.path.splitext(x)[0])
    parts = model_dataframe["part"].unique()
    for n, part in enumerate(parts):
        part = part.split(".")
        parts[n] = part[0]

    results = []

    for counter, part_num in enumerate(parts):
        PART_NUM = part_num
        url = f"https://rebrickable.com/api/v3/lego/parts/{PART_NUM}/"
        headers = {"Authorization": f"key {API_KEY}"}

        while True:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                data_normalized = pd.json_normalize(data)
                results.append(data_normalized)
                break

            elif response.status_code == 404:
                print(f"Fehler Part_NUM: {part_num}, Position: {counter}; nicht gefunden")
                break

            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "3"))
                print(f"Rate Limit erreicht bei {part_num}, Position {counter} → warte {retry_after} Sekunden...")
                time.sleep(retry_after)
                continue

            else:
                print(f"Fehler Part_NUM: {part_num}, Position: {counter}; "
                    f"{response.status_code}; {response.text}")
                break

        time.sleep(1)


    raw_parts_df = pd.concat(results)
    part_category_df = pd.read_csv("./data/part_categories.csv")
    parts_df = raw_parts_df.merge(part_category_df, left_on="part_cat_id", right_on="id", how="left")
    parts_df = parts_df.rename(columns={"name_x": "part_name", "name_y": "category_name"})
    parts_df = parts_df.drop(columns=["id", "part_count"])
    parts_df[["dim1", "dim2", "dim3"]] = parts_df["part_name"].apply(lambda x: pd.Series(extract_dimensions(x)))
    parts_df["bracket_info"] = parts_df["part_name"].apply(lambda x: pd.Series(extract_bracket_info(x)))

    base_name = os.path.splitext(lego_model)[0]
    parts_df.to_csv(os.path.join(processed_data_path, f"{base_name}_parts.csv"), index=False)
    model_dataframe.to_csv(os.path.join(processed_data_path, f"{base_name}_model.csv"), index=False)
    print(f"Model {base_name} finished!")