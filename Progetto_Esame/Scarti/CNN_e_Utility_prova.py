from pathlib import Path
import numpy as np
import cv2 as cv
from keras import layers, Model, Sequential, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# iperparametri globali dell'architettura
# funzione di attivazione (modificabile: 'relu', 'gelu', 'swish')
ACTIVATION_FUNC = 'gelu'

# carica le immagini da una cartella e le ridimensiona a 256x256
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


# divide i dati tra train, validation ed eventuale test, applicando l'one-hot encoding
def divide_and_encode_data(X, y, only_val:bool = False):

    encoder = OneHotEncoder(sparse_output=False)
    
    # uso stratify nello split per mantenere le proporzioni delle classi e non sbilanciare l'addestramento
    if only_val == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_val   = encoder.transform(y_val.reshape(-1, 1))
        class_names = encoder.categories_[0]

        return (X_train, y_train), (X_val, y_val), class_names

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_val   = encoder.transform(y_val.reshape(-1, 1))
        y_test  = encoder.transform(y_test.reshape(-1, 1))

        class_names = encoder.categories_[0]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


def get_data_augmentation():
    return Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),         # rotazioni fino a ±90°, hanno senso per foto aeree o satellitari
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),  # piccoli spostamenti spaziali fino al 10%
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),       
    ], name="data_augmentation")


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

def build_model(input_shape, num_classes):
    """
    Costruisce la rete residuale custom.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Architettura                                                        │
    │                                                                      │
    │  Multi-scale Stem ottimizzato per 256x256 (ResNet-C):                │
    │    Conv 3×3 s=2 → BN → Act   (÷2)                                   │
    │    Conv 3×3 s=1 → BN → Act   (elabora a 128x128)                    │
    │    Conv 3×3 s=1 → BN → Act   (s=1 per ridurre perdita dettagli)     │
    │    MaxPool 3×3 s=2            (÷2)  →  ÷4 totale                    │
    │                                                                      │
    │  Backbone [3, 4, 6, 3] stile ResNet-34 (pre-activation):             │
    │    Stage 1: 3 × ResBlock(64,  stride=1)                             │
    │    Stage 2: 4 × ResBlock(128, stride=2 nel primo)                   │
    │    Stage 3: 6 × ResBlock(256, stride=2 nel primo)                   │
    │    Stage 4: 3 × ResBlock(512, stride=2 nel primo)                   │
    │    BN finale → Act finale  (necessari dopo ultimo blocco pre-act)   │
    │                                                                      │
    │  Riduzione finale:                                                   │
    │    MaxPooling2D(4, s=4)  →  ÷4 aggiuntivo per limitare i parametri   │
    │                                                                      │
    │  Classifier (MLP):                                                   │
    │    Flatten → Dense(1024) → BN → Act → Dropout(0.5)                 │
    │           → Dense(512)  → BN → Act → Dropout(0.5)                  │
    │           → Dense(num_classes, softmax)                             │
    └─────────────────────────────────────────────────────────────────────┘

    Dimensioni spaziali:

      Input      After Stem   S2    S3    S4   After Pool  Flatten
      256×256    64×64        32×32 16×16 8×8  2×2         2 048

    Parametri stimati (~24M con architettura stile ResNet-34 su 256×256).

    Note sul fine-tuning:
      L'input size a 256x256 è usato sia in pre-training che in fine-tuning, 
      permettendo lo slicing diretto del modello senza necessità di reinizializzare 
      i layer Dense.
    """
    inputs = Input(shape=input_shape, name="input")

    #Data augmentation (applicata solo al training set)
    #automaticamente keras disabilita questi layer in fase di inferenza rendendo l'input uguale all'output
    x = get_data_augmentation()(inputs)

    # ── Multi-scale Stem (ResNet-C style ottimizzato per 256x256) ─────────────
    # Tre Conv 3×3 invece della singola Conv 7×7 originale.
    # Abbiamo modificato lo stride della terza conv a 1 per ridurre il downsampling
    # iniziale (da ÷8 a ÷4) per non perdere subito i low-level feature di un 256x256.
    #
    # 256×256 → 128×128 → 128×128 → 128×128 → 64×64  (÷4 totale)
    x = layers.Conv2D(32, 3, strides=2, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    x = layers.Conv2D(32, 3, strides=1, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    # stride modificato da 2 a 1 per preservare i dettagli spaziali
    x = layers.Conv2D(64, 3, strides=1, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # ── Backbone Residuale [3, 4, 6, 3] (Stile ResNet-34 per evitare overfitting) 

    # Stage 1 — 3 blocchi @ 64 filtri, nessun downsampling
    # 256: 64×64
    for i in range(3):
        x = residual_block(x, 64, stride=1)

    # Stage 2 — 4 blocchi @ 128 filtri, stride=2 nel primo
    # 256: 64→32
    x = residual_block(x, 128, stride=2)
    for i in range(3):
        x = residual_block(x, 128, stride=1)

    # Stage 3 — 6 blocchi @ 256 filtri, stride=2 nel primo (ridotti da 12 a 6)
    # 256: 32→16
    x = residual_block(x, 256, stride=2)
    for i in range(5):
        x = residual_block(x, 256, stride=1)

    # Stage 4 — 3 blocchi @ 512 filtri, stride=2 nel primo (ridotti da 6 a 3)
    # 256: 16→8
    x = residual_block(x, 512, stride=2)
    for i in range(2):
        x = residual_block(x, 512, stride=1)

    # Attivazione finale esplicita — necessaria dopo l'ultimo blocco
    # pre-activation (che termina con Add senza Act successiva).
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    # MaxPool finale — riduce la dimensione spaziale di un ulteriore ÷4 per gestire
    # l'aumento di risoluzione dovuto al nuovo Stem, contenendo i parametri del layer Dense.
    # NON è GlobalAveragePooling (vietato dall'assignment): è un MaxPool
    # con kernel 4×4 che porta la mappa da 8x8 a 2x2.
    # 256: 8×8 → 2×2
    x = layers.MaxPooling2D(pool_size=4, strides=4, padding="same")(x)

    # ── Classifier (MLP) ──────────────────────────────────────────────────────
    # Flatten: 256→2×2×512=2048 (Mantenuto costante grazie al MaxPool 4x4)
    x = layers.Flatten()(x)

    x = layers.Dense(1024, name="mlp_dense1")(x)
    x = layers.BatchNormalization(name="mlp_bn1")(x)
    x = layers.Activation(ACTIVATION_FUNC, name="mlp_act1")(x)
    x = layers.Dropout(0.5, name="mlp_drop1")(x)

    x = layers.Dense(512, name="mlp_dense2")(x)
    x = layers.BatchNormalization(name="mlp_bn2")(x)
    x = layers.Activation(ACTIVATION_FUNC, name="mlp_act2")(x)
    x = layers.Dropout(0.5, name="mlp_drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax",name="classifier_head")(x)

    return Model(inputs, outputs)


