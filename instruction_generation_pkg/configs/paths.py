import os

# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONFIG = {
    "paths": {
        # Input-Daten
        "raw": os.path.join(ROOT_DIR, "data", "01_raw"),
        "ready": os.path.join(ROOT_DIR, "data", "02_ready"),
        "features": os.path.join(ROOT_DIR, "data", "03_features"),
        "gt": os.path.join(ROOT_DIR, "data", "ground_truth"),
        "part_categories": os.path.join(ROOT_DIR, "data", "part_categories.csv"),

        # Ergebnisse und Modelle
        "results": os.path.join(ROOT_DIR, "results"),
        "logs": os.path.join(ROOT_DIR, "results", "logs"),
        "models": os.path.join(ROOT_DIR, "results", "models"),
        "plots": os.path.join(ROOT_DIR, "results", "plots"),
    },
}

if __name__ == "__main__":
    print("ROOT_DIR:", ROOT_DIR)
    for k, v in CONFIG["paths"].items():
        print(f"{k}: {v}")
