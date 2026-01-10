import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow_quantum as tfq
import cirq
import sympy
import logging
import numpy as np
from models.qnn_model import create_pqc, create_reuploading_pqc
from config.settings import *
from data.preprocessor import qubits,load_feature_scalers
from data.preprocessor import compute_feature_scalers, preprocess_data, preprocess_data_reuploading

logger = logging.getLogger(__name__)

def build_qnndnn_model():
    """Build the Quantum Neural Network model."""
    pqc, symbols, ops = create_pqc()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string, name='quantum_input'),
        tfq.layers.PQC(pqc, ops, differentiator=tfq.differentiators.Adjoint()),
        tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
                              #kernel_initializer='he_normal',
                              #kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
                              #kernel_initializer='he_normal',
                              #kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
                              #kernel_initializer='he_normal',
                              #kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        # Multiclass
        #tf.keras.layers.Dense(len(class_names), activation='softmax')
        # Binary
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
    
def build_qnndnn_model_with_reuploading_pqc(num_layers=3):
    circuit, data_syms, theta_syms, ops = create_reuploading_pqc(num_layers)

    quantum_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

    pqc_layer = tfq.layers.PQC(circuit, ops, differentiator=tfq.differentiators.Adjoint())

    pqc_output = pqc_layer(quantum_input)

    dense1 = tf.keras.layers.Dense(100, activation="relu")
    drop1  = tf.keras.layers.Dropout(0.2)
    dense2 = tf.keras.layers.Dense(100, activation="relu")
    drop2  = tf.keras.layers.Dropout(0.2)
    dense3 = tf.keras.layers.Dense(100, activation="relu")
    drop3  = tf.keras.layers.Dropout(0.2)

    out = tf.keras.layers.Dense(1, activation="sigmoid")

    h = dense1(pqc_output)
    h = drop1(h)
    h = dense2(h)
    h = drop2(h)
    h = dense3(h)
    h = drop3(h)

    output = out(h)

    model = tf.keras.Model(quantum_input, output)
    return model


def train_qnndnn_model(df, weights_path):
    """Train QNN model on given data."""

    scalers = load_feature_scalers()
    x_train, _ = preprocess_data(df, scalers)
    y_train = df['Label'].values.astype(np.float32)

    model = build_qnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        # Multiclass
        #loss='sparse_categorical_crossentropy',
        # Binary
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
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
                 val_loss=np.array(history.history["val_loss"]),
                 val_accuracy=np.array(history.history["val_accuracy"]), )
    return model

def train_qnndnn_model_reuploading(df, weights_path):
    """Train QNN model on given data."""

    scalers = load_feature_scalers()

    x_train, _ = preprocess_data_reuploading(df, scalers)
    y_train = df['Label'].values.astype(np.int32)

    model = build_qnndnn_model_with_reuploading_pqc()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),#'sparse_categorical_crossentropy',
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
                 val_loss=np.array(history.history["val_loss"]),
                 val_accuracy=np.array(history.history["val_accuracy"]), )
    return model
