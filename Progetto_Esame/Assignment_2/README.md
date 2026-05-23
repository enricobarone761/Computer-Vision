# Assignment 2: Classificazione Immagini Aeree con BoW

## Soluzione Specifica al Problema

### 1-estrai_descrittori.py
**Risolve:** "Calcolo del Vocabolario di Visual Words - Estrazione Descrittori Locali"

- Legge tutte le immagini da `DATASET/AID`
- Estrae keypoint e descrittori **SIFT** da ogni immagine con `cv.SIFT_create(nfeatures=1000)`
- Normalizza i descrittori con **L2** come richiesto
- Salva lista aggregata di tutti i descrittori in `descrittori_e_vacabolario/descrittori_1000.pkl`

### 2-clustering_vocabolario.py
**Risolve:** "Calcolo del Vocabolario - Clustering (K-means)"

- Carica i descrittori estratti
- Applica **KMeans** con tre valori di K richiesti: **{50, 100, 500}**
- I K centroidi = le K "visual words" del vocabolario
- Salva 3 vocabolari: `vocab_k50_1000.pkl`, `vocab_k100_1000.pkl`, `vocab_k500_1000.pkl`

### 3-genera_istogrammi.py
**Risolve:** "Rappresentazione Immagini attraverso Istogrammi BoW"

- Legge tutte le immagini da `DATASET/UCMerced_LandUse` (con etichette di classe)
- Per ogni immagine: estrae descrittori SIFT normalizzati L2
- Assegna ogni descrittore al centroide più vicino con `km_vocabolario.predict()`
- Costruisce istogramma con `np.bincount()`: conta quanti descrittori → ogni visual word
- Normalizza istogrammi con **L2**
- Salva tuple (classe, istogramma) per ogni K: `istogrammi_k50.pkl`, `istogrammi_k100.pkl`, `istogrammi_k500.pkl`

### 4-valutazione_classificatori.py
**Risolve:** "Dataset e Classificatore - Cross-validation"

- Per ogni K ∈ {50, 100, 500}:
  - Estrae X (istogrammi) e y (etichette classi)
  - Applica **3-fold StratifiedKFold** per mantenere bilanciamento delle classi
  - Testa 3 classificatori: **Logistic Regression, Random Forest, SVM (RBF)**
  - Calcola metriche: Accuracy, Precision (macro), Recall (macro), F1-score (macro)
  - Genera **Confusion Matrix** per ogni combinazione (K, classificatore)
- Seleziona miglior modello per K e classificatore in base ad Accuracy
- Addestra il miglior modello su **TUTTO il dataset** senza split
- Salva modello finale: `SVM_k500_addestrato.pkl` (o altro a seconda dei risultati)

### 5-inferenza_manuale.py
**Risolve:** "Inferenza - Predizione su Nuova Immagine"

1. Carica immagine RGB da file
2. Estrae descrittori SIFT normalizzati L2 (**stessa tecnica del training**)
3. Carica vocabolario migliore: `vocab_k500_1000.pkl`
4. Assegna descrittori a visual words con `predict()`
5. Costruisce istogramma BoW normalizzato L2
6. Carica classificatore migliore: `SVM_k500_addestrato.pkl`
7. Predice classe con `classifier.predict(histogram.reshape(1, -1))`

---

**Flusso Complessivo:** AID (descrittori) → Vocabolario → UCMerced (istogrammi) → CV (miglior K e classificatore) → Inferenza
