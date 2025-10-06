# Assembly Instruction Generation via GNN-based Link Prediction

## Zielsetzung
Ziel des Projekts ist die automatische Generierung von Montageanleitungen auf Basis von Bauteildaten.  
Da reale CAD-Daten schwer zugänglich sind, wird für den Prototypen auf **LEGO®-Modelle** zurückgegriffen.  
LEGO eignet sich ideal, da:
- Steckverbindungen klar definiert sind,  
- wiederkehrende Strukturen auftreten,  
- und offene Datenquellen verfügbar sind.

---

## Aufgabenstellung
Gegeben ist ein Satz von LEGO-Bauteilen mit Eigenschaften wie **Position, Rotation, Farbe, Kategorie und Dimensionen**.  
Das Modell soll lernen, **welche Teile miteinander verbunden werden**.  

Dies entspricht einer klassischen **Link-Prediction-Aufgabe** im Kontext von **Graph Neural Networks (GNNs)**.

---

## Modell
- Eingesetzt wird ein **Graph Convolutional Network (GCN)** als Baseline-Modell.  
- Der LEGO-Bausatz wird als Graph dargestellt:  
  - **Knoten:** Bauteile  
  - **Kanten:** physische Verbindungen  
- Der Graph wird mithilfe von `RandomLinkSplit` in **Train/Val/Test** geteilt.  
- Das Modell lernt die Wahrscheinlichkeit einer Verbindung zwischen zwei Knoten zu bestimmen.  

---

## Erstellung der Datenbasis
Da kein bestehender Datensatz vorliegt, wurde ein eigener LEGO-Datensatz erstellt:

1. **Download von Modell-Dateien (.mpd)**  
   - Quelle: [LDraw Official Model Repository (OMR)](https://library.ldraw.org/omr/sets)  
   - Enthält alle Bauteile mit 3D-Positionen.  

2. **Anreicherung über Rebrickable API**  
   - Zusätzliche Metadaten: Kategorie, Dimension, Farbe etc.  

3. **Erstellung der Eingabedaten (Features X)**  
   - Gespeichert in: `df_<modelname>.csv`  

4. **Erstellung der Zielverbindungen (Labels y)**  
   - Gespeichert in: `gt_<modelname>.csv`  
   - Struktur:
     ```
     part_id_1, part_id_2, connected
     123, 456, 1
     789, 101, 0
     ```

---

## Trainingspipeline
- Aufbau des Graphs mit `Data(x, edge_index)`  
- Aufteilung per `RandomLinkSplit` (Train/Val/Test)  
- Training mit:
  - **Loss:** BCEWithLogitsLoss  
  - **Optimizer:** Adam  
  - **Early Stopping:** basierend auf Validation Loss  
- **Evaluation-Metriken:**
  - ROC-AUC  
  - Average Precision (AP)  
  - Hits@K (1, 3, 10, 50)  

---

## Aktueller Stand
- Zwei LEGO-Modelle wurden vollständig verarbeitet und in Graph-Strukturen überführt.  
- Die Trainingspipeline ist implementiert (`Trainer`-Klasse mit fit-, eval- und evaluate-Methoden).  
- Erste Ergebnisse zeigen gute Lernkurven, aber hohe Varianz zwischen Runs → Ursache: zufälliges Sampling.  
- Nächste Schritte:
  1. Erweiterung der Datenbasis um weitere Modelle  
  2. Fixierte Seeds und deterministische Splits  
  3. Modellvergleich (GCN, GraphSAGE, GAT)  
  4. Integration zusätzlicher Features (Farbe, Kategorie, Geometrie)  

---

# Setup & Installation

## Virtual Environment Setup

### Setup (GPU, PyTorch Geometric)
1. Install PyTorch (with your CUDA version, e.g. 12.8):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

2. Install required Libraries:
    pip install -r requirements.txt

3. Get yourself a API-Key from Brickable.com



