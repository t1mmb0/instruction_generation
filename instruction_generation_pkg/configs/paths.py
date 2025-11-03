import os

# ------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONFIG = {
    "paths": {
        # Input-Daten
        "raw_data": os.path.join(ROOT_DIR, "data", "raw"),
        "processed": os.path.join(ROOT_DIR, "data", "processed"),
        "ready": os.path.join(ROOT_DIR, "data", "ready"),

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
