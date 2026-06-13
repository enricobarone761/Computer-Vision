import keras
import Rete_e_Utility as utils

BATCH_SIZE   = 32
EPOCHS       = 150
LR           = 1e-3
PATH = "/home/enrib/progetto/dataset/DATASET/AID" # path per wsl

X, y = utils.load_dataset(PATH)

# split 70/30 solo per il pre-training su AID
(X_train, y_train), (X_val, y_val), class_names = utils.divide_and_encode_data(X, y, only_val=True)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")

model = utils.build_model(input_shape=X.shape[1:], num_classes=len(class_names))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=LR)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/preaddestrato_aid.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir="Progetto_Esame/Assignment_3/logs/strategia1_fase1"
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("Pre-training completato")