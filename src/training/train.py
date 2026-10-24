"""compile + fit (Adam, sparse CE, EarlyStopping)."""
import tensorflow as tf
from keras.callbacks import EarlyStopping

from src.config import LEARNING_RATE, EPOCHS, BATCH_SIZE, ES_PATIENCE


def compile_model(model, learning_rate=LEARNING_RATE):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE, patience=ES_PATIENCE,
                class_weight=None, verbose=1):
    """class_weight=None 이 노트북 최종 설정(SMOTE 사용, class_weight 주석)."""
    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[es],
        verbose=verbose
    )
    return history


def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)
    return test_loss, test_acc
