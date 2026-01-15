# Assembly Instruction Generation via GNN-based Link Prediction

## Zielsetzung
Ziel dieses Projekts ist die automatische Generierung von Montageanleitungen aus strukturierten Bauteildaten.  
Der Fokus liegt auf der Rekonstruktion von Montagegraphen mittels Graph Neural Networks (GNNs).

Da reale CAD-Montagedaten schwer zugänglich sind, dient LEGO als kontrollierte Testumgebung:
- klar definierte Steckverbindungen  
- wiederkehrende Bauteiltypen  
- öffentlich verfügbare 3D- und Metadaten (z. B. LDraw, Rebrickable)

Das Projekt ist forschungsgetrieben und modular aufgebaut, um verschiedene Modelle, Features und Trainingsstrategien systematisch vergleichen zu können.

---

## Aufgabenstellung
Gegeben ist ein Satz von LEGO-Bauteilen mit Eigenschaften wie:
- Position, Orientierung, Farbe, Kategorie und Dimensionen

Aus diesen Informationen soll ein Modell lernen, welche Paare von Bauteilen miteinander verbunden sind.

Dies wird als Link-Prediction-Problem auf einem Graphen formuliert:
- Knoten: Bauteile  
- Kanten: physische Verbindungen  

Das trainierte Modell dient im nächsten Schritt als Score-Funktion, um Montagegraphen iterativ zu konstruieren.

---

## Modellarchitektur (aktueller Stand)

### Encoder
- Baseline: Graph Convolutional Network (GCN)
- Ziel: Lernen von Knoteneinbettungen aus Feature- und Graphstruktur

Weitere Encoder (z. B. GraphSAGE, GAT) sind explizit vorgesehen und können über die Factory-Struktur integriert werden.

### Decoder
- Aktuell: Dot-Product Decoder  
  s(i,j) = z_i^T z_j
- Ausgabe sind Logits, keine Wahrscheinlichkeiten

Der Decoder ist bewusst separat gehalten, um alternative Scoring-Modelle (z. B. MLP- oder bilineare Decoder) einfach testen zu können.

---

## Loss, Optimierung und Regularisierung
- Loss: BCEWithLogitsLoss  
- Optimierer: Adam  
- Learning-Rate-Scheduler: konfigurierbar  
  - epoch-basiert (Cosine, Step, Exponential)  
  - metric-basiert (ReduceLROnPlateau)  
- Early Stopping: val-loss-basiert  
- Weight Decay zur impliziten Regularisierung  

Geplant, aber aktuell nicht implementiert:
- Dropout in Encoder/Decoder  
- Normierung (z. B. LayerNorm)  
- Gradient Clipping  

---

## Framework-Struktur

### GlobalScaler
- Einheitliche Skalierung numerischer Features über mehrere Modelle hinweg  
- Verhindert datenabhängige Skalierungsartefakte  
- Grundlage für stabile Multi-Seed-Experimente

### GraphDataBuilder
- Erzeugt PyTorch-Geometric-Data-Objekte  
- Führt RandomLinkSplit pro Modell aus  
- Unterstützt Multi-Modell-Datasets  
- Bereitet train-, val- und test-Splits für DataLoader vor

### Trainer
- Universeller, DataLoader-basierter Trainer  
- Kernfunktionen:
  - _train_step  
  - _eval_step  
  - _forward_scores  
- Tracking von Train- und Val-Loss  
- Berechnung von ROC-AUC und Average Precision  
- Early Stopping und Best-Model-Reload

### ExperimentRunner und SingleRunExecutor
- Trennung von Experiment-Design, Einzelruns und Training  
- Unterstützung systematischer Multi-Seed-Experimente  
- Ergebnisse werden seedweise gespeichert und vergleichbar gemacht

---

## Datengrundlage

### Feature-Tabellen
df_<model>.csv
```
part_id,color,x,y,z,a,b,c,d,e,f,g,h,i,part,part_name,category_name,dim1,dim2,dim3
```

### Zielkanten
gt_<model>.csv
```
part_id_1, part_id_2, connected
```

---

## Trainingspipeline
1. Globale Feature-Skalierung (GlobalScaler.fit)  
2. Graph-Erstellung und Link-Split (GraphDataBuilder)  
3. Multi-Seed-Training (ExperimentRunner)  
4. Evaluation:
   - ROC-AUC  
   - Average Precision  
   - Loss-Verläufe (Train/Val)  
5. Seed-übergreifende Analyse (Mittelwert und Varianz)

---

## Forschungsvision: Iterativer Graphaufbau
Langfristiges Ziel ist nicht nur die Analyse bestehender Graphen, sondern deren aktive Konstruktion.

Geplanter zweistufiger Ansatz:
1. Training eines Link-Prediction-Modells (aktueller Fokus)
2. Verwendung dieses Modells zur iterativen Konstruktion eines Montagegraphen für neue Modelle:
   - Start mit wenigen Knoten  
   - Vorhersage wahrscheinlicher Kanten  
   - Greedy- oder probabilistisches Hinzufügen  
   - Abbruchkriterien (z. B. Konnektivität oder Score-Schwellen)

Das trainierte Modell fungiert dabei als lokale Entscheidungsfunktion für den Montageprozess.

---

## Aktueller Stand
- Vollständig modularisiertes Trainingsframework  
- Stabile Multi-Seed-Experimente  
- Link Prediction als klar definierte Kernaufgabe  
- Saubere Trennung von Training, Experiment und Evaluation  
- Solide Basis für Modell- und Feature-Exploration  

---

## Nächste Schritte
1. Erhöhung der Trainingsstabilität  
   - kontrolliertes Negative Sampling  
   - stabilere Decoder  
   - Normierung der Embeddings  

2. Modellerweiterungen  
   - alternative Encoder (GraphSAGE, GAT)  
   - stärkere Decoder (MLP, bilinear)

3. Feature-Ablationen  
   - geometrische vs. strukturelle Features  
   - Identitätsmerkmale vs. Generalisierung  

4. Iterativer GraphConstructor  
   - Aufbau-Strategien  
   - Abbruchlogiken  
   - Evaluationsmetriken für Konstruktionsqualität  

---

## Fazit
Dieses Projekt bildet eine forschungsorientierte Grundlage für die Untersuchung,  
wie GNN-basierte Link Prediction zur Rekonstruktion und Konstruktion von Montageprozessen eingesetzt werden kann.

Der Fokus liegt auf Stabilität, Vergleichbarkeit und Modularität,  
um belastbare Aussagen über Modelle, Features und Trainingsstrategien treffen zu können.
