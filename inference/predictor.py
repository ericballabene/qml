import os
import numpy as np
import pandas as pd
import uproot
import logging
import awkward as ak
from config.settings import *
from data.preprocessor import load_feature_scalers, preprocess_data

logger = logging.getLogger(__name__)

def batched_predict(model, x_test, batch_size=512):
    """Run prediction in batches to manage memory."""
    preds = []
    for i in range(0, len(x_test), batch_size):
        batch = x_test[i:i + batch_size]
        preds.append(model.predict(batch, batch_size=len(batch), verbose=0))
    return np.vstack(preds)

def test_and_save(model_A, model_B):
    """Run inference on test data using both models and save results."""
    output_dir = 'output_qnn'
    os.makedirs(output_dir, exist_ok=True)
    scalers = load_feature_scalers()

    for year in years:
        for sample in signal_filenames + background_filenames + data_filenames:
            filepath = os.path.join(base_path, year, sample)
            if not os.path.isfile(filepath):
                logger.warning(f"File {filepath} not found, skipping inference.")
                continue

            logger.info(f"Processing {sample} for year {year} with both models")
            year_dir = os.path.join(output_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            output_filename = os.path.join(year_dir, f"{sample}")

            with uproot.open(filepath) as f:
                if "analysis" not in f:
                    logger.warning(f"No 'reco' tree in {filepath}, skipping.")
                    continue
                tree = f["analysis"]

            if sample in data_filenames:
                variable_to_write = variables_to_copy_data
            else:
                variable_to_write = variables_to_copy

            # Optional safety: keep only branches present in the tree
            available = set(tree.keys())
            variable_to_write = [v for v in variable_to_write if v in available]


            # Collect all chunks for this sample
            all_dfs = []

            iter_source = f"{filepath}:analysis"
            for arrays_chunk in uproot.iterate(iter_source, variable_to_write, library="ak", step_size=50000):
                arrays_chunk = {field: ak.to_numpy(arrays_chunk[field]) for field in arrays_chunk.fields}
                df_chunk = pd.DataFrame(arrays_chunk)
                if df_chunk.empty:
                    continue

                df_A = df_chunk[df_chunk["eventNumber"] % 2 == 0]
                df_B = df_chunk[df_chunk["eventNumber"] % 2 == 1]

                df_processed_all = []

                for model, df_sub in zip([model_B, model_A], [df_A, df_B]):
                    if df_sub.empty:
                        continue
                    x_test, _ = preprocess_data(df_sub, scalers)
                    preds = batched_predict(model, x_test, batch_size=512)

                    # Keep original inputs untouched, add predictions
                    df_out = df_sub.copy()
                    for i, name in enumerate(class_names):
                        df_out[name] = preds[:, i].astype(np.float32)
                    df_processed_all.append(df_out)

                if df_processed_all:
                    all_dfs.append(pd.concat(df_processed_all))

            if not all_dfs:
                continue

            # Global sort across all chunks
            df_final = (
                pd.concat(all_dfs)
                .sort_values("eventNumber")
                .reset_index(drop=True)
            )

            # Convert to numpy arrays for saving
            arrays_to_write = {col: df_final[col].values for col in df_final.columns}

            logger.info(f"Writing output to {output_filename}")
            with uproot.recreate(output_filename) as outf:
                outf["analysis"] = arrays_to_write
