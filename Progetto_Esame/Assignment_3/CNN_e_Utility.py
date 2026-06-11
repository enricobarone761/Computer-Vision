from pathlib import Path
import numpy as np
import cv2 as cv
from keras import layers, Model, Sequential, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


ACTIVATION_FUNC = 'gelu'

# carichiamo le immagini dalla cartella e le ridimensioniamo a 256x256 per uniformarle
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


# data augmentation per combattere un po' l'overfitting
def get_data_augmentation():
    return Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),         # rotazioni fino a ±90°, hanno senso per foto aeree o satellitari
        layers.RandomZoom(0.15),
        #layers.RandomTranslation(0.1, 0.1),  # piccoli spostamenti spaziali fino al 10%
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),       
    ], name="data_augmentation")


def residual_block(x, filters, stride=1):
    """
    Blocco residuo di tipo Pre-Activation.
    
    Flusso dei dati:
      Input -> BN -> Attivazione -> Conv 3x3 -> BN -> Attivazione -> Conv 3x3 -> Somma con Shortcut

    DOpo alcune ricerche ho deciso di implementare la pre-activation come sulla ResNet V2.
    Usare BN e Act prima di ogni convoluzione permette al gradiente di propagarsi più facilmente durante 
    l'addestramento, evitando che i gradienti si appiattiscano all'aumentare dei layer. 
    Inoltre, questo tipo di architettura ha dimostrato di convergere più rapidamente.
    """
    shortcut = x

    # ramo principale
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(filters, 3, strides=stride, padding="same",use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)
    x = layers.Conv2D(filters, 3, strides=1, padding="same",use_bias=False)(x)

    # ── Proiezione shortcut ───────────────────────────────────────────────
    # In un blocco residuo, l'output del "branch principale" viene sommato
    # all'input originale ("shortcut"). Per farlo, devono avere la stessa forma.
    # 
    # Se il branch principale ha modificato le dimensioni (es. rimpicciolendo 
    # l'immagine con stride=2 o aumentando i canali 'filters'), non possiamo 
    # sommarli direttamente. In tal caso applichiamo una Conv2D 1x1 alla shortcut 
    # per forzarla (proiezione) alle nuove dimensioni in modo computazionalmente leggero.
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)

    return layers.Add()([x, shortcut])


def build_model(input_shape, num_classes):
    """
    Architettura                                                        
                                                                        
    Stage 0:                                        
    Conv 3x3 s=2 → BN → Act   (÷2)                                   
    Conv 3x3 s=1 → BN → Act            
    Conv 3x3 s=2 → BN → Act   (÷2)                                   
    MaxPool 3x3 s=2            (÷2)  →  ÷8 totale                    
                                                                        
    Backbone [3, 4, 8, 4] (pre-activation):                           
    Stage 1: 3 x ResBlock(64,  stride=1)                             
    Stage 2: 4 x ResBlock(128, stride=2 nel primo)                   
    Stage 3: 8 x ResBlock(256, stride=2 nel primo)                  
    Stage 4: 4 x ResBlock(512, stride=2 nel primo)                   
    BN finale → Activa finale 
                                                                        
    Riduzione finale:                                                   
    MaxPooling2D(2, s=2)  →  ÷2                           
                                                                        
    Classifier (MLP):                                                   
    Flatten → Dense(1024) → BN → Act → Dropout(0.3)                 
            → Dense(512)  → BN → Act → Dropout(0.3)                  
            → Dense(num_classes, softmax)                             

    Dimensioni spaziali:

    Input      S0       S1     S2     S3     S4    MaxPool    Flatten
    256x256    32x32   32x32  16x16   8x8    4x4      2x2         2048

    Parametri totali: 38.834.824 TODO
    """
    inputs = Input(shape=input_shape, name="input")

    #Data augmentation (applicata solo al training set)
    #automaticamente keras disabilita questi layer in fase di inferenza rendendo l'input uguale all'output
    x = get_data_augmentation()(inputs)

    # ── Multi-scale Stem ─────────────────────────────────────
    # Tre Conv 3x3 
    # La seconda conv a stride=1 processa la feature map a metà risoluzione,
    # estraendo feature prima di scalare ulteriormente.
    #
    # 256×256 → 128×128
    x = layers.Conv2D(32, 3, strides=2, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    # 128×128 → 64×64
    x = layers.Conv2D(32, 3, strides=1, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    # 64×64 → 32×32
    x = layers.Conv2D(64, 3, strides=2, padding="same",use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # ── Backbone Residuale ─────────────────────────────────

    # Stage 1 — 3 blocchi @ 64 filtri
    # 32×32
    for i in range(3):
        x = residual_block(x, 64, stride=1)

    # Stage 2 — 4 blocchi @ 128 filtri, stride=2 nel primo
    # 32x32 → 16x16
    x = residual_block(x, 128, stride=2)
    for i in range(3):
        x = residual_block(x, 128, stride=1)

    # Stage 3 — 8 blocchi @ 256 filtri, stride=2 nel primo
    # 16x16 → 8x8
    x = residual_block(x, 256, stride=2)
    for i in range(7):
        x = residual_block(x, 256, stride=1)

    # Stage 4 — 4 blocchi @ 512 filtri, stride=2 nel primo
    # 8x8 → 4x4
    x = residual_block(x, 512, stride=2)
    for i in range(3):
        x = residual_block(x, 512, stride=1)

    x = layers.BatchNormalization()(x)
    x = layers.Activation(ACTIVATION_FUNC)(x)

    # MaxPool finale — riduce la dimensione spaziale di un ulteriore ÷2.
    # non potendso utiilizzare average pool utilizzo maxpool che
    # con un kernel 2×2 seleziona l'attivazione massima in ogni cella 2×2.
    # 4x4 → 2x2
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(x)

    # ── Classifier (MLP) ──────────────────────────────────────────────────────
    # Flatten: 2x2x512=2048
    x = layers.Flatten()(x)

    x = layers.Dense(1024, name="mlp_dense1")(x)
    x = layers.BatchNormalization(name="mlp_bn1")(x)
    x = layers.Activation(ACTIVATION_FUNC, name="mlp_act1")(x)
    x = layers.Dropout(0.3, name="mlp_drop1")(x)

    x = layers.Dense(512, name="mlp_dense2")(x)
    x = layers.BatchNormalization(name="mlp_bn2")(x)
    x = layers.Activation(ACTIVATION_FUNC, name="mlp_act2")(x)
    x = layers.Dropout(0.3, name="mlp_drop2")(x)

    outputs = layers.Dense(num_classes, activation="softmax",name="classifier_head")(x)

    return Model(inputs, outputs)
