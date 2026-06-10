from pathlib import Path
import numpy as np
import cv2 as cv
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ── Iperparametri globali dell'architettura ───────────────────────────────────
# Funzione di attivazione (modificabile: 'relu', 'gelu', 'swish')
ACTIVATION_FUNC = 'gelu'

# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO E PREPARAZIONE DATI
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cartella):
    X, y = [], []
    for f in Path(cartella).rglob("*"):
        if f.is_file():
            classe = f.parent.name
            im = cv.imread(str(f))
            if im is not None:
                im = cv.resize(im, (256,256))
                X.append(im)
                y.append(classe)
                print(f"Caricata immagine {f} con classe {classe}")
    return np.array(X), np.array(y)


def divide_and_encode_data(X, y, only_val:bool = False):

    encoder = LabelEncoder()
    
    if only_val == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        y_train = encoder.fit_transform(y_train)
        y_val   = encoder.transform(y_val)
        class_names = encoder.classes_

        return (X_train, y_train), (X_val, y_val), class_names

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

        y_train = encoder.fit_transform(y_train)
        y_val   = encoder.transform(y_val)
        y_test  = encoder.transform(y_test)

        class_names = encoder.classes_

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names





def get_data_augmentation():
    #TODO da fare con openCV
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),         # ±90° plausibili per aereo/satellite
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),  # shift spaziali fino al 10%
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),       # variazioni di illuminazione
    ])


# ─────────────────────────────────────────────────────────────────────────────
# BLOCCO RESIDUALE (Pre-Activation, ResNet v2)
# ─────────────────────────────────────────────────────────────────────────────

def residual_block(x, filters, stride=1):
    """
    Pre-activation residual block (He et al., 2016 — ResNet v2).

    Ordine: BN → Act → Conv(3×3) → BN → Act → Conv(3×3) → Add(shortcut)

    Rispetto al blocco post-activation (ResNet v1):
    - Lo shortcut è un'identità pura: i gradienti fluiscono inalterati.
    - La proiezione shortcut usa solo Conv 1×1 senza BN (la BN è già
      applicata sul ramo principale prima della prima conv).
    - Non c'è attivazione dopo l'Add: il prossimo blocco inizia con BN→Act.
    - Nota: l'ultimo blocco della backbone richiede una BN→Act esplicita
      (applicata in build_model prima del MaxPool finale).

    use_bias=False su tutte le Conv seguite da BN (il bias è ridondante
    perché la BN ha il proprio parametro β che svolge la stessa funzione).
    """
    shortcut = x

    # ── Branch principale ─────────────────────────────────────────────────
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(filters, 3, strides=1, padding="same",use_bias=False)(x)

    # ── Proiezione shortcut ───────────────────────────────────────────────
    # Solo se il numero di filtri o la stride cambia.
    # In pre-activation non si aggiunge BN sul percorso shortcut.
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)

    return layers.Add()([x, shortcut])


# ─────────────────────────────────────────────────────────────────────────────
# COSTRUZIONE MODELLO
# ─────────────────────────────────────────────────────────────────────────────

def build_model(input_shape, num_classes, name="Custom_ResNet"):
    """
    Costruisce la rete residuale custom.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Architettura                                                        │
    │                                                                      │
    │  Multi-scale Stem (ResNet-C):                                        │
    │    Conv 3×3 s=2 → BN → Act   (÷2)                                   │
    │    Conv 3×3 s=1 → BN → Act   (elabora a metà risoluzione!)          │
    │    Conv 3×3 s=2 → BN → Act   (÷2)                                   │
    │    MaxPool 3×3 s=2            (÷2)  →  ÷8 totale                    │
    │                                                                      │
    │  Backbone [3, 4, 12, 6] (pre-activation):                           │
    │    Stage 1: 3 × ResBlock(64,  stride=1)                             │
    │    Stage 2: 4 × ResBlock(128, stride=2 nel primo)                   │
    │    Stage 3: 12 × ResBlock(256, stride=2 nel primo)                  │
    │    Stage 4: 6 × ResBlock(512, stride=2 nel primo)                   │
    │    BN finale → Act finale  (necessari dopo ultimo blocco pre-act)   │
    │                                                                      │
    │  Riduzione finale:                                                   │
    │    MaxPooling2D(2, s=2)  →  ÷2 aggiuntivo                          │
    │                                                                      │
    │  Classifier (MLP):                                                   │
    │    Flatten → Dense(1024) → BN → Act → Dropout(0.5)                 │
    │           → Dense(512)  → BN → Act → Dropout(0.5)                  │
    │           → Dense(num_classes, softmax)                             │
    └─────────────────────────────────────────────────────────────────────┘

    Dimensioni spaziali:

      Input      After Stem   S2    S3    S4   After Pool  Flatten
      256×256    32×32        16×16  8×8  4×4  2×2         2 048

    Parametri stimati (~40M con input 256×256).

    Note sul fine-tuning:
      L'input size a 256x256 è usato sia in pre-training che in fine-tuning, 
      permettendo lo slicing diretto del modello senza necessità di reinizializzare 
      i layer Dense.
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # ── Data Augmentation ─────────────────────────────────────────────────────
    x = get_data_augmentation()(inputs)

    # ── Multi-scale Stem (ResNet-C style) ─────────────────────────────────────
    # Tre Conv 3×3 invece della singola Conv 7×7 originale.
    # La seconda conv a stride=1 processa la feature map a metà risoluzione
    # (300×300 per input 600×600), estraendo feature prima di scalare ulteriormente.
    # Questo è il principale vantaggio per immagini ad alta risoluzione.
    #
    # 256×256 → 128×128 → 128×128 →  64×64  → 32×32  (÷8 totale)
    x = layers.Conv2D(32, 3, strides=2, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    x = layers.Conv2D(32, 3, strides=1, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # ── Backbone Residuale ────────────────────────────────────────────────────

    # Stage 1 — 3 blocchi @ 64 filtri, nessun downsampling
    # 600: 75×75  |  256: 32×32
    for i in range(3):
        x = residual_block(x, 64, stride=1)

    # Stage 2 — 4 blocchi @ 128 filtri, stride=2 nel primo
    # 600: 75→37  |  256: 32→16
    x = residual_block(x, 128, stride=2)
    for i in range(3):
        x = residual_block(x, 128, stride=1)

    # Stage 3 — 12 blocchi @ 256 filtri, stride=2 nel primo
    # 600: 37→18  |  256: 16→8
    x = residual_block(x, 256, stride=2)
    for i in range(11):
        x = residual_block(x, 256, stride=1)

    # Stage 4 — 6 blocchi @ 512 filtri, stride=2 nel primo
    # 600: 18→9   |  256: 8→4
    x = residual_block(x, 512, stride=2)
    for i in range(5):
        x = residual_block(x, 512, stride=1)

    # Attivazione finale esplicita — necessaria dopo l'ultimo blocco
    # pre-activation (che termina con Add senza Act successiva).
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    # MaxPool finale — riduce la dimensione spaziale di un ulteriore ÷2.
    # NON è GlobalAveragePooling (vietato dall'assignment): è un MaxPool
    # con kernel 2×2 che seleziona l'attivazione massima in ogni cella 2×2.
    # 600: 9×9 → 4×4  |  256: 4×4 → 2×2
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(x)

    # ── Classifier (MLP) ──────────────────────────────────────────────────────
    # Flatten: 600→4×4×512=8192  |  256→2×2×512=2048
    x = layers.Flatten()(x)

    x = layers.Dense(1024, use_bias=False, name="mlp_dense1")(x)
    x = layers.BatchNormalization(name="mlp_bn1")(x)
    x = layers.Activation(ACTIVATION_FUNC, name="mlp_act1")(x)
    x = layers.Dropout(0.5, name="mlp_drop1")(x)

    x = layers.Dense(512, use_bias=False, name="mlp_dense2")(x)
    x = layers.BatchNormalization(name="mlp_bn2")(x)
    x = layers.Activation(ACTIVATION_FUNC, name="mlp_act2")(x)
    x = layers.Dropout(0.5, name="mlp_drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax",name="classifier_head")(x)

    return keras.Model(inputs, outputs, name=name)


