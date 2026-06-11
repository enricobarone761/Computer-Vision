# Analisi della Rete CNN Residuale Custom

## 1. Panoramica dell'architettura

La rete è una **ResNet custom pre-activation (v2)** con configurazione **[3, 4, 12, 6]** blocchi e uno stem in stile **ResNet-C** (tre Conv 3×3 al posto della singola Conv 7×7).

| Componente | Dettaglio |
|---|---|
| **Stem** | 3 × Conv2D 3×3 + MaxPool 3×3 (÷8 totale) |
| **Backbone** | 25 blocchi residuali pre-activation |
| **Pool finale** | MaxPooling2D 2×2 (÷2 aggiuntivo) |
| **Classifier** | Flatten → Dense(1024) → Dense(512) → Dense(num_classes) |
| **Regolarizzazione** | BN ovunque + Dropout 0.5 nel MLP |
| **Attivazione** | GELU (configurabile) |

---

## 2. Flusso delle dimensioni spaziali (input 256×256)

```
Layer / Stage                  Dimensione       Filtri    Note
─────────────────────────────────────────────────────────────────
Input                          256 × 256 × 3     —
Stem Conv1 (s=2)               128 × 128 × 32    32
Stem Conv2 (s=1)               128 × 128 × 32    32       elabora a mezza ris.
Stem Conv3 (s=2)                64 ×  64 × 64    64
Stem MaxPool (s=2)              32 ×  32 × 64    64       ÷8 totale dallo stem
Stage 1 (×3, s=1)               32 ×  32 × 64    64
Stage 2 (×4, primo s=2)         16 ×  16 × 128  128
Stage 3 (×12, primo s=2)         8 ×   8 × 256  256
Stage 4 (×6, primo s=2)          4 ×   4 × 512  512
BN + Act finale                  4 ×   4 × 512  512
MaxPool finale (s=2)             2 ×   2 × 512  512       ÷2 aggiuntivo
Flatten                         2048              —        2×2×512
Dense(1024)                     1024              —
Dense(512)                       512              —
Dense(num_classes)              num_classes        —
```

> [!NOTE]
> Con input 600×600 (originale non ridimensionato), il vettore dopo Flatten sarebbe 4×4×512 = **8192** invece di **2048**. Questo quadruplica i parametri del primo layer Dense (da ~2M a ~8.4M). Il resize a 256 è quindi una scelta pragmatica sensata per il training.

---

## 3. Stima parametri

### Backbone (Conv + BN)

| Stage | Blocchi | Filtri in → out | Parametri Conv (stimati) | Note |
|---|---|---|---|---|
| Stem | — | 3→32→32→64 | ~28 K | 3 Conv 3×3 |
| Stage 1 | 3 | 64→64 | ~3 × 73K ≈ **221 K** | Nessuna proiezione |
| Stage 2 | 4 | 64→128 (primo), 128→128 | ~8K (proj) + 73K + 3×295K ≈ **966 K** | Proiezione 1×1 nel primo blocco |
| Stage 3 | 12 | 128→256 (primo), 256→256 | ~33K (proj) + 295K + 11×1.18M ≈ **13.3 M** | Blocco dominante |
| Stage 4 | 6 | 256→512 (primo), 512→512 | ~131K (proj) + 1.18M + 5×4.72M ≈ **24.9 M** | Più parametri/blocco |
| **Totale backbone** | | | **~39.4 M** | |

### Classifier (MLP)

| Layer | Parametri |
|---|---|
| Dense(1024) | 2048 × 1024 + 1024 = **~2.1 M** |
| Dense(512) | 1024 × 512 + 512 = **~524 K** |
| Dense(21) | 512 × 21 + 21 = **~10.8 K** |
| BN (×2) | ~3 K |
| **Totale MLP** | **~2.6 M** |

### Totale complessivo stimato: **~42 M parametri**

> [!IMPORTANT]
> La rete ha circa **42 milioni di parametri**. È un modello di dimensioni paragonabili a una ResNet-50 (~25.5M) ma più profondo nella parte centrale (12 blocchi nello Stage 3). I 6 blocchi finali a 512 filtri sono il componente più pesante.

---

## 4. Analisi delle scelte architetturali

### ✅ Cose fatte bene

1. **Stem ResNet-C**: Tre Conv 3×3 → campo recettivo equivalente a una 7×7 ma con **meno parametri** e **più non-linearità**. Scelta consolidata (usata in ResNet-D, ResNeXt, etc.).

2. **Pre-activation (ResNet v2)**: L'ordine BN → Act → Conv garantisce che lo **shortcut sia un'identità pura**, il che migliora il flusso dei gradienti nei modelli profondi. Scelta corretta per una rete con 25 blocchi.

3. **BN + Act finale esplicita** (riga 202-203): Necessaria e correttamente implementata. Nell'ultimo blocco pre-activation, l'output esce dall'Add senza attivazione — la BN+Act esplicita chiude correttamente la catena.

4. **`use_bias=False`** su tutte le Conv seguite da BN: Corretto, il bias è ridondante con il parametro β della BatchNorm.

5. **Data augmentation**: Inserita come layer del modello → automaticamente disattivata in inference. Rotazioni fino a ±90° sono appropriate per immagini aeree/satellitari (l'orientamento è arbitrario).

6. **Dropout 0.5**: Aggressivo ma giustificato per un modello di questa taglia con dataset relativamente piccoli (AID ~10K immagini, UC Merced ~2.1K).

7. **Naming convention `mlp_`**: Facilita il fine-tuning selettivo (come in [Strategia1_Fase2_Opt1.py](file:///c:/Users/enrib/Documents/Ing_Informatica_LM32/Computer-Vision/Progetto_Esame/Assignment_3/Strategia1_Fase2_Opt1.py#L42)).

### ⚠️ Punti di attenzione e potenziali criticità

#### 4.1 — Resize 600→256: impatto sulla perdita di informazione

Le immagini originali sono 600×600. Il resize a 256×256 comporta un **fattore di riduzione ~5.5x nell'area** (da 360K pixel a 65K pixel).

Per immagini aeree, dove i dettagli fini contano (es. storage tanks, tennis courts, sparse buildings), questa riduzione può essere significativa. Tuttavia:

- Il dataset AID ha **10.000 immagini a 600×600**, e il resize riduce enormemente l'uso di RAM (~3.5 GB per AID a 256 vs ~13 GB a 600, in float32).
- Il resize è un **trade-off ragionevole** per il pre-training dove il volume di dati compensa parzialmente la perdita di dettaglio.

> [!TIP]
> Se le risorse lo consentono, potresti valutare un resize a **384×384** come compromesso: mantieni ~4x più pixel rispetto a 256 e i tempi di training aumentano di ~2.25x. Le dimensioni spaziali restano compatibili con la rete (384 ÷ 16 = 24, tutte le divisioni per 2 funzionano senza problemi di arrotondamento).

#### 4.2 — Commenti interni dello stem con dimensioni "600"

Nei commenti alla riga 157 si legge:

```python
# (300×300 per input 600×600)
```

E nella tabella dimensioni dello Stage alla riga 178:

```
# 600: 75×75  |  256: 32×32
```

Questi commenti si riferiscono al caso con input originale 600×600. **Sono corretti come documentazione**, ma dato che le immagini vengono **sempre** passate a 256×256 (anche nel pre-training, vedi `load_dataset` alla riga 20), la colonna "600" nei commenti descrive uno scenario che non viene mai eseguito nel codice attuale. Non è un bug, ma può confondere chi legge.

#### 4.3 — Proiezione shortcut nel blocco pre-activation

Alla [riga 98](file:///c:/Users/enrib/Documents/Ing_Informatica_LM32/Computer-Vision/Progetto_Esame/Assignment_3/CNN_e_Utility.py#L98):

```python
shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
```

La proiezione viene applicata sull'**input originale `x`** (prima di BN+Act). Nella letteratura pre-activation originale (He et al. 2016), ci sono due varianti:
- **Variante A** (la tua): proiezione sullo shortcut grezzo → semplice e funziona bene in pratica.
- **Variante B**: proiezione dopo BN+Act → teoricamente più coerente.

La tua implementazione è la variante più comune e va bene. Nota solo che lo shortcut bypassa la prima BN+Act, il che introduce un lieve disallineamento di scala. In pratica, l'effetto è trascurabile.

#### 4.4 — MaxPool finale vs GlobalMaxPool

Usi `MaxPooling2D(2, 2)` alla riga 209 che porta la dimensione a 2×2. Il Flatten produce poi 2048 valori. Questo è conforme al vincolo dell'assignment (no average pooling). Tuttavia:

- Un **GlobalMaxPooling2D** produrrebbe un vettore di 512 valori indipendentemente dalla risoluzione di input → renderebbe la rete **agnostica alla risoluzione**.
- Con il tuo design attuale, se cambi la risoluzione di input (es. 384×384), il Flatten produrrà 3×3×512 = 4608 e **i layer Dense non saranno compatibili** con i pesi pre-addestrati a 256.

> [!WARNING]
> L'assignment vieta **average pooling** ma NON vieta **max pooling globale**. Un `GlobalMaxPooling2D` renderebbe la rete più flessibile per il fine-tuning a risoluzioni diverse. Con il design attuale sei **vincolato a usare 256×256 anche nel fine-tuning**, il che è gestibile ma limita le opzioni.

#### 4.5 — RAM: caricamento interamente in memoria

`load_dataset()` carica **tutte** le immagini in un singolo `np.array`. Per AID a 256×256:

```
10.000 immagini × 256 × 256 × 3 bytes = ~1.88 GB (uint8)
                                        ~7.5 GB (float32 dopo normalizzazione implicita)
```

Questo funziona su macchine con 16+ GB di RAM, ma potrebbe essere al limite. Se riscontri problemi di memoria, un `tf.data.Dataset` con caricamento lazy sarebbe più robusto. Per UC Merced (~2100 immagini) non è un problema.

#### 4.6 — Normalizzazione degli input

Non vedo un passaggio esplicito di **normalizzazione pixel** (es. `X = X / 255.0` o `X = X.astype('float32') / 255.0`). Il range [0, 255] uint8 viene passato direttamente alla rete.

> [!CAUTION]
> **Verifica se la normalizzazione avviene altrove** (negli script di training). Se il modello riceve valori [0, 255] interi, la BatchNorm nello stem può comunque adattarsi, ma:
> - La convergenza sarà **più lenta** rispetto a input normalizzati in [0, 1].
> - I gradienti iniziali saranno più grandi del necessario.
> - La data augmentation (RandomBrightness, RandomContrast) potrebbe comportarsi diversamente su range diversi.
>
> **Ti consiglio fortemente di aggiungere `X = X.astype('float32') / 255.0`** dopo il caricamento, oppure un `layers.Rescaling(1./255)` come primo layer prima della data augmentation.

---

## 5. Profondità e campo recettivo

Con 25 blocchi residuali (2 Conv 3×3 ciascuno = 50 Conv) + 3 Conv nello stem = **53 layer convoluzionali**, la rete è paragonabile in profondità a una **ResNet-101** (senza bottleneck).

Il campo recettivo teorico a 256×256 è molto ampio (copre facilmente l'intera immagine grazie ai 4 livelli di downsampling ÷2 nel backbone + ÷8 nello stem = **÷128 totale**), il che è appropriato per la classificazione di scene aeree dove il contesto globale è determinante.

---

## 6. Riepilogo e raccomandazioni

| Aspetto | Valutazione | Priorità |
|---|---|---|
| Architettura complessiva | ✅ Solida e ben progettata | — |
| Pre-activation blocks | ✅ Implementazione corretta | — |
| Stem ResNet-C | ✅ Scelta moderna ed efficace | — |
| Data augmentation | ✅ Appropriata per foto aeree | — |
| **Normalizzazione input** | ❌ Potenzialmente mancante | **Alta** |
| MaxPool vs GlobalMaxPool | ⚠️ Limita la flessibilità di risoluzione | Media |
| Commenti "600×600" | ⚠️ Potenzialmente confondenti | Bassa |
| Caricamento in memoria | ⚠️ Può saturare la RAM su AID | Media |
| Resize 256 | ✅ Compromesso ragionevole | — |

### Raccomandazioni concrete:

1. **🔴 Aggiungi normalizzazione** — `X = X.astype('float32') / 255.0` in `load_dataset()` o un `layers.Rescaling(1./255)` nella rete.
2. **🟡 Valuta GlobalMaxPooling2D** — al posto di MaxPool + Flatten per rendere la rete agnostica alla risoluzione.
3. **🟢 Pulisci i commenti "600"** — o aggiungi una nota che indica che 600 è la dimensione originale non usata.
