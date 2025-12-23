import json
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from config.settings import *

logger = logging.getLogger(__name__)

qubits = [cirq.GridQubit(0, i) for i in range(NUM_QUBITS)]

def encode_features_as_circuit(feature_values):
    """Encode feature values as quantum circuit using RX gates."""
    if len(feature_values) != NUM_QUBITS:
        raise ValueError(f"Expected {NUM_QUBITS} features, got {len(feature_values)}")
    circuit = cirq.Circuit()
    for i, val in enumerate(feature_values):
        circuit.append(cirq.rx(float(val))(qubits[i]))
    return circuit

def compute_feature_scalers(df, scaler_path=scaler_file):
    """Compute min/max scalers for all features and save to JSON."""
    scalers = {}
    for f in features:
        fmin, fmax = float(df[f].min()), float(df[f].max())
        scalers[f] = {"min": fmin, "max": fmax}
    with open(scaler_path, "w") as fp:
        json.dump(scalers, fp)
    return scalers

def load_feature_scalers(scaler_path=scaler_file):
    """Load saved feature scalers from JSON."""
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file {scaler_path} not found.")
    with open(scaler_path, "r") as fp:
        return json.load(fp)

def apply_scalers(df, scalers):
    """Apply scaling to features and normalize to [0, π] range."""
    df = df.copy()
    feature_cols = []
    for f in features:
        newcol = f + "_training_"
        fmin, fmax = scalers[f]["min"], scalers[f]["max"]
        if fmax == fmin:
            df[newcol] = 0.0
        else:
            # 1) scale to [0, 1]
            x = (df[f] - fmin) / (fmax - fmin)
            # 2) clip outliers
            x = np.clip(x, 0.0, 1.0)
            # 3) map to [-π, +π]
            df[newcol] = (x * 2 - 1) * np.pi
        feature_cols.append(newcol)
    return df, feature_cols

def preprocess_data(df, scalers):
    """Preprocess data: scale features and convert to quantum circuits."""
    df, feature_cols = apply_scalers(df, scalers)
    circuits = [encode_features_as_circuit(vals) for vals in df[feature_cols].values]
    circuit_tensor = tfq.convert_to_tensor(circuits)
    return circuit_tensor, df
