import keras
import CNN_e_Utility as utils

BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 1e-3
PATH = "/home/enrib/progetto/dataset/DATASET/UCMerced_LandUse/Images"

X, y = utils.load_dataset(PATH)

(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = utils.divide_and_encode_data(X, y)

print(f"Training set: {X_train.shape[0]} campioni")
print(f"Validation set: {X_val.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

model = utils.build_model(input_shape=X_train.shape[1:], num_classes=len(class_names))
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=7,
        restore_best_weights=True, 
        verbose=1),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=5,
        min_lr=1e-6, 
        verbose=1),

    keras.callbacks.ModelCheckpoint(
        filepath="Progetto_Esame/Assignment_3/Modelli_e_CF/strategia2.keras",
        monitor="val_loss", 
        save_best_only=True, 
        verbose=1),

    keras.callbacks.TensorBoard(
        log_dir="Progetto_Esame/Assignment_3/logs/strategia2", 
        histogram_freq=0
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