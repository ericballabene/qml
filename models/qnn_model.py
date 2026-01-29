import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy

from models.pqc import (
    create_pqc,
    create_reuploading_pqc,
    create_reuploading_pqc_with_multiqubit_correlators,
    create_reuploading_pqc_with_alltoallentanglement_multiqubit_correlators,
)

from config.settings import *
from data.preprocessor import (
    load_feature_scalers,
    preprocess_data,
    preprocess_data_reuploading,
)

from utils.seed import set_global_determinism

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# GLOBAL SEEDING
# ---------------------------------------------------------------------
SEED = 42
set_global_determinism(SEED)

# ---------------------------------------------------------------------
# OPTIONAL IMPROVEMENTS
# ---------------------------------------------------------------------

# Deterministic PQC parameter initialization
# PQC_INITIALIZER = tf.keras.initializers.GlorotUniform(seed=SEED)

# Better classical head initialization
# DENSE_INITIALIZER = tf.keras.initializers.HeNormal(seed=SEED)

# Enable BatchNorm after PQC (NOT equivalent)
ENABLE_BATCH_NORM = False

# Extra metrics (does not affect training)
ENABLE_AUC_METRIC = False

# Disable XLA for determinism
DISABLE_JIT = False

# ---------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------
def build_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
        ),
    ]

# ---------------------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------------------
def build_qnn_model():
    pqc, symbols, ops = create_pqc()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(), dtype=tf.string, name="quantum_input"),
            tfq.layers.PQC(
                pqc,
                ops,
                differentiator=tfq.differentiators.Adjoint(),
                # IMPROVEMENT (disabled):
                # initializer=PQC_INITIALIZER,
            ),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                # IMPROVEMENT (disabled):
                # kernel_initializer=DENSE_INITIALIZER,
            ),
        ]
    )
    return model


def build_qnn_model_with_reuploading(pqc_builder, num_layers=3):
    circuit, data_syms, theta_syms, ops = pqc_builder(num_layers)

    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

    x = tfq.layers.PQC(
        circuit,
        ops,
        differentiator=tfq.differentiators.Adjoint(),
        # IMPROVEMENT (disabled):
        # initializer=PQC_INITIALIZER,
    )(circuit_input)

    if ENABLE_BATCH_NORM:
        x = tf.keras.layers.BatchNormalization()(x)

    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        # IMPROVEMENT (disabled):
        # kernel_initializer=DENSE_INITIALIZER,
    )(x)

    return tf.keras.Model(inputs=circuit_input, outputs=output)

# ---------------------------------------------------------------------
# GENERIC TRAIN FUNCTION
# ---------------------------------------------------------------------
def train(
    df,
    weights_path,
    output_dir,
    model_builder,
    preprocess_fn,
    batch_size=128,
    epochs=50,
):
    os.makedirs(output_dir, exist_ok=True)

    weights_file = os.path.join(output_dir, weights_path)
    history_file = os.path.join(
        output_dir, weights_path.replace(".h5", ".npz")
    )

    scalers = load_feature_scalers()
    x_train, _ = preprocess_fn(df, scalers)
    y_train = df["Label"].values.astype(np.int32)

    model = model_builder()

    metrics = ["accuracy"]
    if ENABLE_AUC_METRIC:
        metrics.append(tf.keras.metrics.AUC(name="auc"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
        # IMPROVEMENT (disabled):
        # jit_compile=not DISABLE_JIT,
    )

    logger.info(model.summary())

    if os.path.isfile(weights_file):
        logger.info(f"Loading existing weights from {weights_file}")
        model.load_weights(weights_file)
        return model

    logger.info(f"Training new model → {weights_file}")

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=build_callbacks(),
        verbose=2,
    )

    model.save_weights(weights_file)

    np.savez(
        history_file,
        loss=history.history["loss"],
        accuracy=history.history.get("accuracy"),
        val_loss=history.history["val_loss"],
        val_accuracy=history.history.get("val_accuracy"),
        auc=history.history.get("auc"),
        val_auc=history.history.get("val_auc"),
    )

    return model

# ---------------------------------------------------------------------
# TRAINING APIS
# ---------------------------------------------------------------------
def train_qnn_model(df, weights_path, output_dir="UNK"):
    return train(
        df,
        weights_path,
        output_dir,
        model_builder=build_qnn_model,
        preprocess_fn=preprocess_data,
    )


def train_qnn_model_reuploading(df, weights_path, output_dir="UNK"):
    return train(
        df,
        weights_path,
        output_dir,
        model_builder=lambda: build_qnn_model_with_reuploading(
            create_reuploading_pqc, num_layers=3
        ),
        preprocess_fn=preprocess_data_reuploading,
    )


def train_qnn_model_reuploading_with_multiqubit_correlators(
    df, weights_path, output_dir="UNK"
):
    return train(
        df,
        weights_path,
        output_dir,
        model_builder=lambda: build_qnn_model_with_reuploading(
            create_reuploading_pqc_with_multiqubit_correlators, num_layers=3
        ),
        preprocess_fn=preprocess_data_reuploading,
    )


def train_qnn_model_reuploading_with_alltoallentanglement_multiqubit_correlators(
    df, weights_path, output_dir="UNK"
):
    return train(
        df,
        weights_path,
        output_dir,
        model_builder=lambda: build_qnn_model_with_reuploading(
            create_reuploading_pqc_with_alltoallentanglement_multiqubit_correlators,
            num_layers=3,
        ),
        preprocess_fn=preprocess_data_reuploading,
        batch_size=128,
    )

