import os
import pandas as pd
import uproot
import numpy as np
import logging
from config.settings import (
    features, years, signal_totrain_filenames, background_filenames,
    base_path, preselection_training, skim_vars, columns_to_load
)

logger = logging.getLogger(__name__)

def get_label(sample_name):
    """Map sample names to labels."""
    if sample_name == "Signal.root":
        return 1
    else:
        return 0  # background

signal_labels = sorted({get_label(s) for s in signal_totrain_filenames})

def enforce_types_automatic(arrays: pd.DataFrame) -> pd.DataFrame:
    """Automatically ensure type consistency across ROOT branches."""
    for col in arrays.columns:
        if arrays[col].dtype == "object":
            # Flatten jagged arrays
            if arrays[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                arrays[col] = arrays[col].apply(lambda x: x[0] if len(x) > 0 else np.nan)
            # Try numeric conversion
            arrays[col] = pd.to_numeric(arrays[col], errors='ignore')

    # After flattening, make all numeric columns consistent float64
    numeric_cols = arrays.select_dtypes(include=['number']).columns
    arrays[numeric_cols] = arrays[numeric_cols].astype('float64')

    return arrays

def read_training_samples_limited(max_per_signal=3000, max_background_total=3000, apply_selection=True):
    """Read and limit training samples with type consistency enforced."""
    background_label = 0
    collected_signal = {lbl: 0 for lbl in signal_labels}
    background_count = 0
    dfs = []

    for year in years:
        for sample in signal_totrain_filenames + background_filenames:
            label = get_label(sample)

            # Skip if quota reached
            if label != background_label and collected_signal[label] >= max_per_signal:
                continue
            if label == background_label and background_count >= max_background_total:
                continue

            filepath = os.path.join(base_path, year, sample)
            if not os.path.isfile(filepath):
                logger.warning(f"File {filepath} not found, skipping.")
                continue

            with uproot.open(filepath) as f:
                if "analysis" not in f:
                    logger.warning(f"Tree 'analysis' not found in {filepath}, skipping.")
                    continue

                tree = f["analysis"]
                arrays = tree.arrays(columns_to_load, library="pd")
                if arrays.empty:
                    continue

                if apply_selection:
                    arrays = arrays.query(preselection_training)
                    if arrays.empty:
                        continue

                # Drop skim variables
                arrays = arrays.drop(columns=[c for c in skim_vars if c in arrays.columns])

                # Add sample info
                arrays["Sample"] = sample
                arrays["Label"] = label

                # Enforce automatic type consistency
                arrays = enforce_types_automatic(arrays)

                # Apply quota logic
                if label == background_label:
                    remaining_for_file = max_background_total // len(background_filenames)
                    remaining = min(remaining_for_file, max_background_total - background_count)
                    if remaining <= 0:
                        continue
                    if len(arrays) > remaining:
                        arrays = arrays.sample(n=remaining, random_state=42)
                    background_count += len(arrays)
                else:
                    needed = max_per_signal - collected_signal[label]
                    if needed <= 0:
                        continue
                    if len(arrays) > needed:
                        arrays = arrays.sample(n=needed, random_state=42)
                    collected_signal[label] += len(arrays)

                dfs.append(arrays)

                # Early exit if quotas satisfied
                if (
                    all(v >= max_per_signal for v in collected_signal.values()) and
                    background_count >= max_background_total
                ):
                    df_train = pd.concat(dfs, ignore_index=True)
                    logger.info("Training sample counts:\n%s", df_train.groupby(["Sample", "Label"]).size())
                    return df_train

    if not dfs:
        raise RuntimeError("No training data passed selection.")

    df_train = pd.concat(dfs, ignore_index=True)
    logger.info("Training sample counts:\n%s", df_train.groupby(["Sample", "Label"]).size())
    return df_train

def split_AB(df: pd.DataFrame):
    """Split data into A and B subsets based on eventNumber."""
    A = df[df['eventNumber'] % 2 == 0].copy()
    B = df[df['eventNumber'] % 2 == 1].copy()
    return A, B
