import json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, MODEL_PATH,
    CLASS_NAMES_PATH, IMAGE_SIZE, BATCH_SIZE, SEED, EPOCHS
)


def load_datasets():
    for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not folder.exists():
            raise FileNotFoundError(
                f"{folder} does not exist. Run `python split_dataset.py` first."
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    if class_names != val_ds.class_names or class_names != test_ds.class_names:
        raise ValueError("Train, validation and test folders must contain the same classes.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def build_model(number_of_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plots(history):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "loss_curve.png")
    plt.close()


def main():
    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Classes:", class_names)

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_training_plots(history)

    # Load the best checkpoint before final test evaluation.
    best_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        probabilities = best_model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probabilities.argmax(axis=1).tolist())

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\nClassification report:\n", report)

    with open(MODEL_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved class names: {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
