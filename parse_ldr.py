import pandas as pd

def parse_ldr_to_df(filepath):
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
