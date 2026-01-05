import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import logging
import numpy as np
from config.settings import *
from data.preprocessor import qubits,load_feature_scalers
from data.preprocessor import compute_feature_scalers, preprocess_data, preprocess_data_reuploading

logger = logging.getLogger(__name__)

def create_pqc():
    """Create Parameterized Quantum Circuit."""
    symbols = sympy.symbols(f'theta0:{NUM_QUBITS}')
    circuit = cirq.Circuit()
    for i, q in enumerate(qubits):
        circuit.append(cirq.ry(symbols[i])(q))
    # Simple chain entanglement
    for i in range(NUM_QUBITS - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    readout_operators = [cirq.Z(q) for q in qubits]
    return circuit, symbols, readout_operators

def create_reuploading_pqc(num_layers=3):
    data_symbols = sympy.symbols(f'x0:{NUM_QUBITS}')
    theta_symbols = sympy.symbols(
        f'theta0:{num_layers * NUM_QUBITS}'
    )

    circuit = cirq.Circuit()
    theta_idx = 0

    for layer in range(num_layers):
        # --- Data encoding ---
        for i, q in enumerate(qubits):
            circuit.append(cirq.rx(data_symbols[i])(q))

        # --- Trainable block ---
        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(theta_symbols[theta_idx])(q))
            theta_idx += 1

        # --- Entanglement ---
        for i in range(NUM_QUBITS - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    readout_operators = [cirq.Z(q) for q in qubits]
    return circuit, data_symbols, theta_symbols, readout_operators

def build_qnn_model():
    pqc, symbols, ops = create_pqc()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string, name='quantum_input'),

        # Quantum feature extractor
        tfq.layers.PQC(pqc, ops, differentiator=tfq.differentiators.Adjoint()),

        # Classical classifier head
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_qnn_model_with_reuploading_pqc(num_layers=3):
    circuit, data_syms, theta_syms, ops = create_reuploading_pqc(num_layers)

    # Input is now just string tensor of circuits
    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

    pqc_output = tfq.layers.PQC(
        circuit,
        ops,
        differentiator=tfq.differentiators.Adjoint()
    )(circuit_input)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(pqc_output)

    model = tf.keras.Model(inputs=circuit_input, outputs=output)
    return model


def train_qnn_model(df, weights_path):
    """Train QNN model on given data."""

    scalers = load_feature_scalers()

    x_train, _ = preprocess_data(df, scalers)
    y_train = df['Label'].values.astype(np.int32)

    model = build_qnn_model()
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



def train_qnn_model_reuploading(df, weights_path):
    """Train QNN model on given data."""

    scalers = load_feature_scalers()

    x_train, _ = preprocess_data_reuploading(df, scalers)
    y_train = df['Label'].values.astype(np.int32)

    model = build_qnn_model_with_reuploading_pqc()
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
