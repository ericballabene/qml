import tensorflow as tf
import numpy as np
import os
import logging
from config.settings import *
from data.preprocessor import load_feature_scalers, compute_feature_scalers, preprocess_data_dnn

logger = logging.getLogger(__name__)

def build_dnn_model(input_shape):
    """Build a standard DNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
        tf.keras.layers.Dropout(0.2),
        # Binary
        tf.keras.layers.Dense(1, activation='sigmoid')
        # Multiclass
        #tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model

def train_dnn_model(df, weights_path):
    """Train DNN model on given data."""

    scalers = load_feature_scalers()
    x_train, _ = preprocess_data_dnn(df, scalers)
    # int for binary
    y_train = df['Label'].values.astype(np.int32)

    model = build_dnn_model(x_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        #loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)
    ]

    if os.path.isfile(weights_path):
        logger.info(f"Loading existing weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        logger.info(f"Training new model and saving weights to {weights_path}")
        history = model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=2, validation_split=0.2, callbacks=callbacks)
        model.save_weights(weights_path)
        np.savez(weights_path.replace(".h5", ".npz"),
                 loss=np.array(history.history["loss"]),
                 accuracy=np.array(history.history["accuracy"]),
                 val_loss=np.array(history.history.get("val_loss", [])),
                 val_accuracy=np.array(history.history.get("val_accuracy", [])))
    return model
