import os
import numpy as np
import uproot
import tensorflow as tf
import pandas as pd
from config.settings import (
    years,
    signal_filenames,
    background_filenames,
    data_filenames,
    base_path,
    variables_to_copy, variables_to_copy_data
)
from data.loader import split_AB

def batched_predict(vae, X, batch_size=128):
    """Compute VAE losses in batches."""
    RecoLoss_list, KL_list, Loss_list = [], [], []

    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        x_hat, mean, log_var = vae(batch)
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        mse = tf.reduce_mean((batch - x_hat) ** 2, axis=1).numpy()
        kl = (-0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)).numpy()
        loss = mse + kl
        RecoLoss_list.append(mse)
        KL_list.append(kl)
        Loss_list.append(loss)

    RecoLoss = np.concatenate(RecoLoss_list)
    KL = np.concatenate(KL_list)
    Loss = np.concatenate(Loss_list)
    return RecoLoss, KL, Loss

def run_vae_losses(X_scaled, vae):
    """Compute VAE losses for a scaled dataset."""
    return batched_predict(vae, X_scaled, batch_size=8)

def test_and_save(vae_A, vae_B, feature_names, scaler_A, scaler_B, output_dir="output_vae"):
    """Run VAE inference per-year, per-sample, saving ROOT files with variables_to_copy + VAE losses."""
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        for sample in signal_filenames + background_filenames + data_filenames:
            filepath = os.path.join(base_path, year, sample)
            if not os.path.isfile(filepath):
                print(f"File {filepath} not found, skipping.")
                continue

            year_dir = os.path.join(output_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            output_file = os.path.join(year_dir, sample)

            # --- Read variables_to_copy from ROOT ---
            with uproot.open(filepath) as f:
                if "analysis" not in f:
                    print(f"No 'analysis' tree in {filepath}, skipping.")
                    continue
                tree = f["analysis"]

                if sample in data_filenames:
                    variable_to_write = variables_to_copy_data
                else:
                    variable_to_write = variables_to_copy

                available = set(tree.keys())
                variable_to_write = [v for v in variable_to_write if v in available]

                arrays_copy = {}
                for v in variable_to_write:
                    if v not in tree.keys():
                        raise ValueError(f"Variable {v} not found in {filepath}")
                    arrays_copy[v] = tree[v].array(library="np")

                # Build X_sample from features
                X_arrays = [arrays_copy[f] for f in feature_names]
                X_sample = np.column_stack(X_arrays)

            # --- Split into A/B using split_AB ---
            df_sample = pd.DataFrame(X_sample, columns=feature_names)
            df_sample["eventNumber"] = arrays_copy["eventNumber"]
            A_df, B_df = split_AB(df_sample)

            # --- Scale separately ---
            X_A_scaled = scaler_A.transform(A_df[feature_names].to_numpy())
            X_B_scaled = scaler_B.transform(B_df[feature_names].to_numpy())

            # --- Compute VAE losses ---
            RecoLoss_A, KL_A, Loss_A = run_vae_losses(X_A_scaled, vae_A)
            RecoLoss_B, KL_B, Loss_B = run_vae_losses(X_B_scaled, vae_B)

            # --- Merge back to full order ---
            VAE_RecoLoss = np.zeros(len(X_sample))
            VAE_KL       = np.zeros(len(X_sample))
            VAE_Loss     = np.zeros(len(X_sample))

            VAE_RecoLoss[A_df.index] = RecoLoss_A
            VAE_KL[A_df.index]       = KL_A
            VAE_Loss[A_df.index]     = Loss_A

            VAE_RecoLoss[B_df.index] = RecoLoss_B
            VAE_KL[B_df.index]       = KL_B
            VAE_Loss[B_df.index]     = Loss_B

            # --- Prepare arrays for ROOT ---
            arrays_to_write = {v: arrays_copy[v].astype(np.float64).flatten() for v in variable_to_write}
            arrays_to_write.update({
                "VAE_RecoLoss": VAE_RecoLoss.astype(np.float64),
                "VAE_KL": VAE_KL.astype(np.float64),
                "VAE_Loss": VAE_Loss.astype(np.float64),
            })

            # --- Write ROOT tree ---
            with uproot.recreate(output_file) as f_out:
                f_out["analysis"] = arrays_to_write

            print(f"Saved ROOT for {sample} ({year}): {output_file}")

