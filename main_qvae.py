import os
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.settings import features
from utils.helpers import setup_logging, setup_reproducibility
from data.loader import read_training_samples_limited, split_AB
from models.quantum_vae import train_on_subset
from inference.qvae_inference import test_and_save

OUTPUT_DIR = "output_qvae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    setup_reproducibility()

    # === READ AND PREPARE DATA ===
    df = read_training_samples_limited(
        max_per_signal=0, max_background_total=10000
    )
    df = df.sample(frac=1.0, random_state=47).reset_index(drop=True)

    # Split into A/B subsets
    A, B = split_AB(df)

    # Extract numeric features using the already filtered list from config
    X_full = df[features].to_numpy(dtype=np.float32)
    X_A = A[features].to_numpy(dtype=np.float32)
    X_B = B[features].to_numpy(dtype=np.float32)

    # === FEATURE SCALING PER SUBSET ===
    scaler_A = StandardScaler()
    X_A_scaled = scaler_A.fit_transform(X_A)

    scaler_B = StandardScaler()
    X_B_scaled = scaler_B.fit_transform(X_B)

    # === TRAIN OR LOAD VAE MODELS ===
    vae_A = train_on_subset(
        X_A_scaled,
        latent_dim=3,
        epochs=100,
        model_name="A",
        save_dir=OUTPUT_DIR
    )

    vae_B = train_on_subset(
        X_B_scaled,
        latent_dim=3,
        epochs=100,
        model_name="B",
        save_dir=OUTPUT_DIR
    )

    test_and_save(
        vae_A,
        vae_B,
        feature_names=features,
        scaler_A=scaler_A,
        scaler_B=scaler_B,
        output_dir=OUTPUT_DIR
    )

    logger.info("VAE training and per-sample inference completed!")


if __name__ == "__main__":
    main()
