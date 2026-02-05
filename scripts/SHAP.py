import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import tensorflow_quantum as tfq
import uproot
import json
import logging

# -------------------------
# Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from config.settings import features, NUM_QUBITS, class_names
from data.preprocessor import apply_scalers, encode_features_as_circuit
from models.pqc import (create_reuploading_pqc_with_alltoallentanglement_multiqubit_correlators)
from models.qnn_model import build_qnn_model_with_reuploading

# -------------------------
# Paths
# -------------------------
OUTPUT_QNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output_qnn_batchsize32/"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output_qnn_batchsize32/"))
SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../feature_scalers.json"))
OUTPUT_PDF_SUMMARY = os.path.join("shap_output", "shap_summary.pdf")
OUTPUT_PDF_BEESWARM = os.path.join("shap_output", "shap_beeswarm.pdf")

YEARS = ['2015-16']

saved_model_A = os.path.join(OUTPUT_QNN_DIR, "QNN_A.h5")
saved_model_B = os.path.join(OUTPUT_QNN_DIR, "QNN_B.h5")

# -------------------------
# Helper Functions
# -------------------------
def load_feature_scalers(path=None):
    if path is None:
        path = SCALER_PATH
    with open(path, "r") as fp:
        scalers = json.load(fp)
    logger.info(f"Loaded scalers for features: {list(scalers.keys())}")
    return scalers

def split_AB(df):
    df_sorted = df.sort_values("eventNumber").reset_index(drop=True)
    A = df_sorted[df_sorted["eventNumber"] % 2 == 0].copy()
    B = df_sorted[df_sorted["eventNumber"] % 2 == 1].copy()
    logger.info(f"Split dataset: A={len(A)}, B={len(B)}")
    return A, B

def preprocess_for_shap(df, scalers, features):
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.error(f"Missing feature columns in DataFrame: {missing}")
        raise RuntimeError(f"Missing feature columns in DataFrame: {missing}")

    # Apply scalers
    df_scaled, feature_cols = apply_scalers(df, scalers)

    # Remove '_training_' suffix
    feature_cols = [f.replace("_training_", "") for f in feature_cols]

    x_numeric = df_scaled[feature_cols].to_numpy(dtype=float)

    logger.info(f"Preprocessed {len(feature_cols)} features: {feature_cols}")
    logger.info(f"x_numeric shape: {x_numeric.shape}")
    return x_numeric, df_scaled, feature_cols

def make_predict_fn_for_model(model, feature_cols):
    def predict_fn(x_numpy):
        x_arr = np.asarray(x_numpy)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        circuits = [encode_features_as_circuit(row) for row in x_arr]
        circuits_tensor = tfq.convert_to_tensor(circuits)
        preds = model.predict(circuits_tensor)

        if isinstance(preds, tf.Tensor):
            preds = preds.numpy()
        # Ensure shape is (num_samples, num_outputs)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        logger.debug(f"Predictions shape: {preds.shape}")
        return preds
    return predict_fn

def compute_shap(model, df, scalers, features, nsamples=100):
    x_background, df_scaled, feature_cols = preprocess_for_shap(df, scalers, features)
    predict_fn = make_predict_fn_for_model(model, feature_cols)

    # Check shapes before SHAP
    logger.info(f"x_background shape: {x_background.shape}")

    explainer = shap.KernelExplainer(predict_fn, x_background, link="identity")
    sample_data = df_scaled.sample(n=min(nsamples, len(df_scaled)), random_state=42)
    x_sample = sample_data[feature_cols].to_numpy(dtype=float)

    # Extra check
    if x_sample.shape[1] != len(feature_cols):
        raise RuntimeError("Mismatch between sample data columns and feature_cols!")

    shap_values = explainer.shap_values(x_sample)

    # If single-output model, shap_values may be a list with one element
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]

    logger.info(f"SHAP values shape: {np.array(shap_values).shape}")
    return shap_values, feature_cols, df_scaled, x_sample

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("shap_output", exist_ok=True)

    # Load scalers
    scalers = load_feature_scalers()

    # Build and load model A
    logger.info("Building QNN architecture for model A")
    model_A = build_qnn_model_with_reuploading(
        create_reuploading_pqc_with_alltoallentanglement_multiqubit_correlators,
        num_layers=3,
    )
    model_A.load_weights(saved_model_A)

    # Build and load model B
    logger.info("Building QNN architecture for model B")
    model_B = build_qnn_model_with_reuploading(
        create_reuploading_pqc_with_alltoallentanglement_multiqubit_correlators,
        num_layers=3,
    )
    model_B.load_weights(saved_model_B)

    # -------------------------
    # Load data
    # -------------------------
    dfs = []
    for year in YEARS:
        year_dir = os.path.join(OUTPUT_DIR, year)
        if not os.path.isdir(year_dir):
            logger.warning(f"Year directory does not exist: {year_dir}")
            continue

        for fname in os.listdir(year_dir):
            if not fname.endswith(".root"):
                continue

            file_path = os.path.join(year_dir, fname)
            try:
                with uproot.open(file_path) as f:
                    if "analysis;1" not in f:
                        logger.warning(f"'analysis;1' tree not found in {fname}")
                        continue

                    reco = f["analysis;1"]
                    reco_keys = reco.keys()
                    columns_to_load = features + ["eventNumber"] + class_names
                    available_cols = [c for c in columns_to_load if c in reco_keys]

                    if not available_cols:
                        logger.warning(f"No requested columns in {fname}. Requested: {columns_to_load}")
                        logger.warning(f"Available columns: {list(reco_keys)}")
                        continue

                    n_entries = min(100, reco.num_entries)
                    df = reco.arrays(available_cols, library="pd", entry_stop=n_entries)

                    if df.empty:
                        logger.warning(f"Empty DataFrame from {fname}")
                        continue

                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} events from {fname} with columns {available_cols}")

            except Exception as e:
                logger.warning(f"Error reading {fname}: {e}")

    if not dfs:
        raise RuntimeError("No valid data to compute SHAP.")

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total events loaded: {len(df_all)}")
    logger.info(f"Columns in combined DataFrame: {list(df_all.columns)}")

    # Split into A/B
    A_df, B_df = split_AB(df_all)
    A_sample = A_df.sample(n=min(50, len(A_df)), random_state=42)
    B_sample = B_df.sample(n=min(50, len(B_df)), random_state=42)

    # -------------------------
    # Compute SHAP
    # -------------------------
    logger.info("Computing SHAP: Model A on B_sample")
    shap_values_A_on_B, feature_cols, _, x_B_sample = compute_shap(model_A, B_sample, scalers, features, nsamples=100)

    logger.info("Computing SHAP: Model B on A_sample")
    shap_values_B_on_A, feature_cols, _, x_A_sample = compute_shap(model_B, A_sample, scalers, features, nsamples=100)
    

    # Average SHAP values
    shap_values_avg = 0.5 * (shap_values_A_on_B + shap_values_B_on_A)
    shap_values_avg = np.squeeze(shap_values_avg, axis=-1)

    #print("Final feature columns for SHAP plot:", feature_cols)
    logger.info(f"SHAP values shape: {shap_values_avg.shape}")

    # -------------------------
    # Plot SHAP summary
    # -------------------------
    df_plot = pd.DataFrame(x_B_sample, columns=feature_cols)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_avg,
        df_plot,
        feature_names=feature_cols,
        show=False,
        plot_type="bar"
    )
    plt.savefig(OUTPUT_PDF_SUMMARY, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,6))
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values_avg,
            data=df_plot.values,
            feature_names=feature_cols
        ),
        show=False
    )
    plt.savefig(OUTPUT_PDF_BEESWARM, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary saved to {OUTPUT_PDF_SUMMARY} and {OUTPUT_PDF_BEESWARM}")

if __name__ == "__main__":
    main()

