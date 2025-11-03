# Assembly Instruction Generation via GNN-based Link Prediction

## 🎯 Zielsetzung
Ziel des Projekts ist die **automatische Generierung von Montageanleitungen** auf Basis von CAD-ähnlichen Bauteildaten.  
Da reale CAD-Daten schwer zugänglich sind, wird als Prototyp eine **LEGO-basierte Testumgebung** verwendet.  

LEGO bietet ideale Voraussetzungen:
- klar definierte Steckverbindungen,  
- wiederkehrende Strukturen,  
- öffentlich verfügbare 3D-Datenquellen (z. B. LDraw, Rebrickable).

---

## 🧩 Aufgabenstellung
Gegeben ist ein Satz von LEGO-Bauteilen mit Eigenschaften wie:
- **Position, Rotation, Farbe, Kategorie und Dimensionen.**

Das Zielmodell soll lernen, **welche Teile miteinander verbunden werden**, um daraus eine **Bauabfolge** zu rekonstruieren.  
Dies entspricht einer klassischen **Link-Prediction-Aufgabe** im Kontext von **Graph Neural Networks (GNNs)**.

Langfristig wird der Graph **nicht nur analysiert**, sondern **iterativ aufgebaut** — ähnlich wie beim realen LEGO-Bauprozess.

---

## 🧠 Modellarchitektur
- Aktuelles Basismodell: **Graph Convolutional Network (GCN)**  
- Repräsentation des LEGO-Modells als Graph:
  - **Knoten:** Bauteile  
  - **Kanten:** physische Verbindungen  
- Aufteilung mit `RandomLinkSplit` in **Train/Val/Test**  
- **Ziel:** Vorhersage der Wahrscheinlichkeit P(edge=True | x_i, x_j)

### 🔹 Loss & Optimierung
- **Loss:** `BCEWithLogitsLoss`  
- **Optimizer:** `Adam`  
- **Scheduler:** `ReduceLROnPlateau`  
- **Regularisierung:** Dropout, BatchNorm, Gradient Clipping

---

## ⚙️ Framework-Struktur

### 1️⃣ GlobalScaler
- Vereinheitlicht Feature-Skalierung über mehrere Modelle hinweg  
- Identifiziert numerische Features und füllt fehlende Werte  
- Ermöglicht stabile, modellübergreifende Trainingsdaten

### 2️⃣ GraphDataBuilder
- Baut PyTorch-Geometric-kompatible Graph-Objekte (`Data`)  
- Unterstützt Multi-Modell-Training  
- Führt `RandomLinkSplit` pro Modell aus  
- Bereitet `train`, `val`, `test`-Listen für DataLoader vor  

### 3️⃣ Trainer
- Universeller, DataLoader-basierter Trainer:
  - `_train_step()`, `_eval_step()`, `_forward_scores()`  
  - Tracking von Loss-Verläufen (Train/Val)  
  - ROC-AUC & Average Precision als Standardmetriken  
- Unterstützt GPU-Training, Early-Stopping & Checkpointing  
- Inferenz & Analyse über `_forward_scores()`

### 4️⃣ Iterativer Graph-Aufbau (in Entwicklung)
- **Ziel:** Rekonstruktion eines Modells durch schrittweises Hinzufügen von Kanten  
- Greedy oder probabilistische Strategien:
  - Auswahl der wahrscheinlichsten Verbindung  
  - Hinzufügen zum aktuellen Graph-Zustand  
- **Abbruchkriterien:**  
  - Alle Teile mindestens einmal verbunden  
  - Graph ist zusammenhängend  
  - Durchschnittlicher Knotengrad über Schwelle  
  - Keine weiteren Kanten mit Score > Threshold  

---

## 📊 Erstellung der Datenbasis
1. **Download der LDraw-Modelle**  
   Quelle: [LDraw Official Model Repository (OMR)](https://library.ldraw.org/omr/sets)

2. **Anreicherung über Rebrickable API**  
   - Zusatzinfos: Kategorie, Jahr, Dimension  

3. **Feature-Extraktion (DataFrame df_<model>.csv)**
   ```
   part_id,color,x,y,z,a,b,c,d,e,f,g,h,i,part,part_name,category_name,dim1,dim2,dim3
   ```

4. **Zielkanten (Labels gt_<model>.csv)**
   ```
   part_id_1, part_id_2, connected
   ```

5. **Graph-Erzeugung**
   - Aus `df_*.csv` → Knotenmerkmale  
   - Aus `gt_*.csv` → Zielkanten  

---

## 📈 Trainingspipeline
1. **Feature-Skalierung** (`GlobalScaler.fit()`)  
2. **Graph-Erstellung & Split** (`GraphDataBuilder`)  
3. **Trainingsphase** (`Trainer.fit(train_loader, val_loader)`)  
4. **Evaluation:**  
   - ROC-AUC  
   - Average Precision  
   - Lernkurven (Train/Val-Loss)  
5. **Testphase & Analyse:**  
   - `Trainer.evaluate_model(test_loader)`  
   - Score-Verteilungen, Fehlanalysen  

---

## 🧱 Iterativer Aufbau (Zielrichtung)
Das trainierte Link-Prediction-Modell dient als **Score-Engine** zur Graphkonstruktion:
1. Start mit zwei zufälligen Teilen  
2. Berechne Verbindungswahrscheinlichkeiten  
3. Füge höchste Wahrscheinlichkeiten als Kanten hinzu  
4. Wiederhole, bis Abbruchbedingungen erfüllt sind  

→ So entsteht ein **autonom wachsender Graph**, der den realen Bauprozess simuliert.

---

## 🔍 Aktueller Stand
- Framework vollständig modularisiert (`GlobalScaler`, `GraphDataBuilder`, `Trainer`)  
- Trainer läuft stabil auf mehreren Modellen via DataLoader  
- ROC-AUC & Average Precision implementiert  
- Loss-Verläufe werden aufgezeichnet (`trainer.history`)  
- Grundlagen für **iterative Graph-Konstruktion** und **stabile Trainingsprozesse** gelegt  

---

## 🧭 Nächste Schritte
1. **Trainingsstabilität erhöhen**
   - Early-Stopping, Scheduler, Gradient Clipping  
   - Seed-Fixierung, Loss-Glättung  

2. **Feature-Engineering erweitern**
   - Geometrische, strukturelle und farbbasierte Merkmale integrieren  

3. **Auswertung & Analyse**
   - Lernkurven, ROC-Kurven, Fehlermuster, Embedding-Visualisierung  

4. **Iterativer GraphConstructor**
   - Greedy-/Top-k-Aufbau, Abbruchkriterien, Feedback-Loop  

5. **Vergleich & Erweiterung**
   - Weitere Modelle: GraphSAGE, GAT, GIN  
   - Stabilitätsmetriken (z. B. AUC-Varianz über Runs)

---

## ⚙️ Setup & Installation

### GPU-Setup mit PyTorch Geometric
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.txt
```

### Zusätzliche Schritte
- API-Key von [Rebrickable.com](https://rebrickable.com/api) anlegen  
- Daten in `results/` ablegen (`df_*.csv`, `gt_*.csv`)  

---

## 🧩 Fazit
Dieses Framework bildet die Grundlage für eine **iterative, GNN-basierte Montageanleitungs-Generierung**.  
Es kombiniert klassische Link Prediction mit einem **dynamischen Aufbauprozess**,  
der reale **Bauabläufe (z. B. LEGO)** modelliert und Schritt für Schritt einen **Graphen konstruiert**,  
statt nur bestehende Verbindungen zu erkennen.
