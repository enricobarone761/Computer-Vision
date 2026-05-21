## --- PAGINA 1 ---

*(Nella parte superiore della pagina è presente una griglia di 21 immagini satellitari/aeree quadrate a colori, che mostrano diverse tipologie di territorio e infrastrutture come campi agricoli, aerei in pista, spiagge, foreste, autostrade, parcheggi, zone residenziali, serbatoi di stoccaggio e campi da tennis).*

**Assignment 2 - Classificazione di Immagini attraverso BoW**

Sviluppare un sistema completo per la classificazione di immagini aeree acquisite da camere RGB.

Data una immagine in input, il sistema dovrà classificare l'immagine come appartenente ad una delle seguenti 21 classi: agricultural, airplane, baseball_diamond, beach, buildings, chaparral, dense_residential, forest, freeway, golf_course, harbor, intersection, medium_residential, mobile_homepark, overpass, parking_lot, river, runway, sparse_residential, storage_tanks, tennis_court.

Il metodo dovrà utilizzare un approccio Bag-of-Visual-Words in cui un vocabolario di $K$ "parole visuali" è addestrato a partire dai descrittori locali (es. SIFT o ORB). Quindi, ogni immagine è rappresentata attraverso un istogramma di tali visual words. Un opportuno classificatore shallow (per es. SVM o Random Forest) è quindi addestrato per classificare ogni immagine in una delle 21 classi.

### 1. Calcolo del Vocabolario di Visual Words

> Il vocabolario è un insieme di $K$ "parole visuali" (centroidi) apprese da descrittori locali estratti da un insieme di immagini (dataset AID).

#### Descrizione del Diagramma di Flusso: Pipeline di estrazione del vocabolario

Il diagramma mostra 5 blocchi sequenziali connessi da frecce:

1. **1 - RACCOLTA IMMAGINI (Dataset AID):** Vengono mostrate delle miniature di immagini aeree. Testo sotto: *"Si seleziona un insieme di immagini dal dataset AID. Le etichette NON vengono utilizzate."*
2. **2 - ESTRAZIONE DESCRITTORI LOCALI:** Un'immagine aerea presenta dei cerchi gialli sui punti chiave (keypoint). Accanto ci sono dei grafici a barre orizzontali blu che rappresentano i vettori dei descrittori. Testo sotto: *"Per ogni immagine si rilevano keypoint (es. SIFT o ORB) e si calcolano i relativi descrittori. Ogni riga blu rappresenta un descrittore."*
3. **3 - RACCOLTA E NORMALIZZAZIONE DEI DESCRITTORI:** Viene mostrata una grande matrice formata dall'unione dei grafici a barre blu. Testo esplicativo: *"Si uniscono tutti i descrittori estratti da tutte le immagini in una grande matrice $X \in \mathbb{R}^{N \times D}$ (dove $N = \text{numero totale di descrittori}$, $D = \text{dimensione del descrittore}$). È consigliata una normalizzazione (es. L2)."*
4. **4 - CLUSTERING (K-MEANS):** Un grafico cartesiano bidimensionale mostra dei punti raggruppati in cluster di diversi colori (verde, blu, giallo, rosso, azzurro, viola), ognuno con una "X" nera al centro che indica il centroide. Testo sotto: *"Si applica k-means con K cluster sul set di descrittori X. Ogni 'X' rappresenta un centroide (una visual word)."*
5. **5 - VOCABOLARIO DI VISUAL WORDS:** Viene mostrato un elenco di elementi ($W_1, W_2, W_3, \dots, W_K$), ognuno associato a un istogramma a barre colorate. Testo sotto: *"I K centroidi ottenuti formano il vocabolario di visual words. Ogni centroide rappresenta una 'parola visuale'."*

---

**Specifiche del calcolo del vocabolario:**
Utilizzare opencv per l'estrazione di descrittori locali (SIFT o ORB, a vostra scelta) dalle immagini.

Per apprendere il vocabolario di visual words, useremo un dataset diverso da quello usato per l'addestramento del classificatore. In particolare, si usino le immagini del dataset AID (Aerial Image Dataset) che può essere scaricato dal link alla pagina:
[https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets).

Le etichette del dataset AID non devono essere utilizzate: le immagini servono esclusivamente per estrarre descrittori locali.

Siete liberi di scegliere se limitare o meno il numero di immagini o di descrittori estratti in base alle risorse di calcolo/memoria a vostra disposizione (se estraete troppi descrittori locali e non riuscite a gestirli, potete sempre sotto-campionare i descrittori. Potreste decidere di limitare il numero di descrittori estratti a 100-200 per immagine o comunque a 100k per l'intero dataset).

Applicare ai descrittori, se lo si reputa utile, una normalizzazione appropriata (L1 o L2).

Una volta che un buon numero di descrittori è disponibile, apprendere un vocabolario di parole visuali tramite clustering. Si usi k-means con $K$ prefissato. I $K$ centroidi rappresenteranno approssimazioni delle visual words e formeranno il vocabolario.

È utile salvare i descrittori o il vocabolario sfruttando la libreria `pickle` (per evitare di dover ripetere i calcoli ogni volta).
**Attenzione:** il vocabolario va costruito una sola volta all'inizio.

---

## --- PAGINA 2 ---

### 2. Rappresentazione delle immagini attraverso istogrammi di Visual Words

**COME SI USA IL VOCABOLARIO PER COSTRUIRE L'ISTOGRAMMA DI UN'IMMAGINE**

> Dato un vocabolario di $K$ visual words (centroidi), ogni immagine viene rappresentata come un istogramma di $K$ elementi.

#### Descrizione del Diagramma di Flusso: Costruzione dell'istogramma

Il diagramma descrive il processo diviso in 5 passi:

1. **1 - IMMAGINE DI INPUT:** Viene mostrata un'immagine aerea quadrata. Testo sotto: *"Immagine da classificare"*.
2. **2 - ESTRAZIONE DESCRITTORI LOCALI:** L'immagine presenta cerchietti gialli sui keypoint. A destra compaiono i singoli vettori estratti contrassegnati come $d_1, d_2, d_3, \dots, d_M$. Testo sotto: *"Si rilevano i keypoint (es. SIFT o ORB) e si calcolano i relativi descrittori. Ogni riga blu rappresenta un descrittore (vettore)"*.
3. **3 - ASSEGNAZIONE AI CENTROIDI PIÙ VICINI:** Delle frecce collegano i vettori $d_i$ allo spazio dei cluster (visto a pagina 1). Ogni vettore viene associato al centroide geometricamente più vicino. Testo sotto: *"Ogni descrittore è assegnato al centroide più vicino (nearest centroid)"*.
4. **4 - CONTEGGIO DELLE OCCORRENZE:** Tabella con l'elenco delle visual words e il conteggio di quanti descrittori le hanno scelte (Es. $W_1 = 5$, $W_2 = 12$, $W_3 = 3$, $\dots$, $W_K = 7$). Testo sotto: *"Si conta quante volte i descrittori sono stati assegnati a ciascuna visual word"*.
5. **5 - ISTOGRAMMA DI VISUAL WORDS:** Un grafico a barre in cui l'asse verticale indica la "frequenza" e l'asse orizzontale mostra le etichette $W_1, W_2, W_3, \dots, W_K$. Testo sotto: *"L'istogramma (di dimensione K) rappresenta l'immagine ed è usato come feature vector per il classificatore"*.

---

Fissato il vocabolario, costituito da $K$ parole visuali, possiamo rappresentare ogni immagine attraverso un istogramma di $K$ elementi.

Per ogni immagine, estraiamo i descrittori locali, assegniamo ciascun descrittore alla parola visuale del vocabolario più vicina (nearest centroid), costruiamo un istogramma delle visual words. È consigliato normalizzare gli istogrammi (L1 o L2).

### 3. Dataset e Classificatore

**Come si usano i descrittori delle immagini per addestrare il classificatore**

> Ogni immagine viene rappresentata come un istogramma delle visual words (Bag-of-Words). Questi istogrammi, insieme alle etichette di classe, vengono usati per addestrare un classificatore supervisionato.

#### Descrizione del Diagramma di Flusso: Pipeline di addestramento

Il diagramma mostra l'integrazione dei dati per la compilazione della matrice di addestramento:

1. **1 - ESTRAZIONE DESCRITTORI (da ogni immagine):** Miniature di tre immagini con i rispettivi vettori/istogrammi estratti. Testo sotto: *"Da ogni immagine si estraggono keypoint (es. SIFT/ORB) e i relativi descrittori locali (vettori)"*.
2. **2 - COSTRUZIONE ISTOGRAMMI (Bag-of-Words):** Rappresentazione degli istogrammi affiancati per ogni immagine sfruttando il Vocabolario a $K$ parole. Testo sotto: *"Ogni immagine è rappresentata da un istogramma di dimensione K. Per ogni immagine: assegnazione dei descrittori alla visual word più vicina e conteggio delle occorrenze"*.
3. **3 - MATRICE DEI DESCRITTORI (FEATURE MATRIX):** Gli istogrammi vengono impilati come righe in una tabella/matrice di dimensioni $[ \text{N\_immagini} \times K ]$, dove le colonne sono le visual words $W_1, W_2, W_3, \dots, W_K$ e i valori interni sono i conteggi numerici (es. la prima riga contiene valori come 5, 12, 3, 7).
4. **4 - VETTORE DELLE ETICHETTE (CLASSI):** Un vettore colonna parallelo alla matrice che contiene le etichette testuali delle classi corrispondenti a ciascuna immagine (es. agricultural, airplane, beach, harbor, ..., tennis_court). Testo sopra: *"Per ogni immagine è nota l'etichetta di classe"*.
5. **5 - ADDESTRAMENTO DEL CLASSIFICATORE:** Un grafico mostra una linea di decisione (iperpiano) che separa i campioni nello spazio delle feature (Feature 1 vs Feature 2). Testo sopra: *"Si utilizza la matrice delle feature X e il vettore delle etichette y per addestrare un classificatore shallow (es. SVM, Random Forest, k-NN, ecc.)"*.

---

Utilizzeremo l'**UC Merced Land Use Dataset** disponibile al link:
[https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset).

Si utilizzi un **3-fold cross-validation protocol** avendo cura di lasciare il numero di immagini per ciascuna classe bilanciato.

Utilizzare almeno due classificatori shallow (a vostra scelta) tra i seguenti: SVM lineare, SVM con kernel RBF, Random Forest, k-NN, Logistic Regression.

Potete tunare, se volete, gli iperparametri dei classificatori per ogni fold ricavando un validation set dal training set di ogni iterazione della cross-validation (a dire il vero, suggerisco di usare gli iper-parametri di default per semplicità e di mantenerli costanti in ogni fold).

### 4. Valutazione del metodo

Utilizzare metriche appropriate per la classificazione multi-classe: Accuracy, Precision, Recall, F1-score (macro-average), Confusion Matrix.
Le metriche possono essere calcolate tramite scikit-learn (`classification_report`, `confusion_matrix`).

Gli esperimenti in cross-validation devono permettere di:

* studiare il comportamento del sistema al variare della dimensione del vocabolario $K$ in $\{50, 100, 500\}$ e la selezione del valore più conveniente di $K$ (**obbligatorio**)
* selezionare il miglior classificatore tra i 2 (almeno) scelti (**obbligatorio**)
* selezionare le migliori features locali tra SIFT e ORB (**opzionale**)

### 5. Inferenza

Prevedere uno script/una funzione che permetta all'utente di utilizzare il miglior modello su una nuova immagine. Dato il path di una immagine RGB in input, il sistema dovrà:

1. caricare l'immagine;
2. estrarre i descrittori locali usando la stessa tecnica scelta in fase di training;
3. assegnare i descrittori alle visual words del vocabolario appreso;
4. costruire l'istogramma BoW dell'immagine;

---

## --- PAGINA 3 ---

5. applicare la stessa eventuale normalizzazione usata in training;
6. usare il miglior classificatore selezionato in cross-validation per predire la classe dell'immagine.

È quindi richiesto di salvare e riutilizzare tutti gli elementi necessari all'inferenza, ad esempio: vocabolario di visual words; eventuali parametri di normalizzazione; miglior classificatore addestrato.

**FASE DI INFERENZA: come predire la classe di una nuova immagine**

> Dato il vocabolario ($K$ visual words) e il miglior classificatore addestrato, il sistema predice la classe di una nuova immagine.

#### Descrizione del Diagramma di Flusso: Pipeline di Inferenza

Il diagramma lineare descrive i passaggi logici che compie la funzione di inferenza su un file esterno:

1. **1 - INPUT: NUOVA IMMAGINE:** Viene inserita un'immagine RGB (mostrata come miniatura quadrata).
2. **2 - ESTRAZIONE DESCRITTORI (stessa tecnica del training):** Vengono individuati i keypoint rilevati (cerchi gialli) e sotto viene visualizzato un grafico continuo dei descrittori locali.
3. **3 - ASSEGNAZIONE AL VOCABOLARIO (nearest centroid):** I vettori estratti ($d_1, d_2, d_3, \dots, d_M$) vengono mappati all'interno delle regioni di decisione dei cluster del vocabolario pre-calcolato.
4. **4 - COSTRUZIONE ISTOGRAMMA BoW:** Viene generato l'istogramma di frequenza grezzo (grafico a barre con asse verticale "Frequenza" e asse orizzontale con le $K$ bin: $W_1, W_2, W_3, \dots, W_K$).
5. **5 - NORMALIZZAZIONE (opzionale ma consigliata):** L'istogramma precedente viene riscalato. Viene mostrato un grafico analogo con la dicitura *"Istogramma normalizzato"*.
6. **6 - PREDIZIONE CON IL MIGLIOR CLASSIFICATORE:** L'istogramma normalizzato viene inviato al modello finale. Viene mostrato il grafico spaziale delle feature in cui il nuovo elemento viene posizionato e classificato in base ai confini appresi. Testo esplicativo: *"Si usa il miglior classificatore selezionato in cross-validation (es. SVM lineare, RBF, Random Forest,...)"*.

---

### Presentazione dei risultati e Consegna

Oltre alla consegna del codice, è richiesta una breve presentazione in formato slide (PDF).

La presentazione deve contenere al **massimo 10 slide** (esclusa una slide aggiuntiva finale) e deve descrivere in modo chiaro e sintetico:

* l'approccio utilizzato (pipeline del metodo)
* le scelte progettuali principali (feature, vocabolario, classificatori)
* eventuali problemi riscontrati durante lo sviluppo e come sono stati affrontati
* i risultati ottenuti in cross-validation e il modello selezionato
* una matrice di confusione relativa al miglior modello
* possibili miglioramenti del sistema