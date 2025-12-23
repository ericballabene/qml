import logging
import tensorflow as tf
import numpy as np

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

def setup_reproducibility():
    """Set random seeds for reproducibility."""
    tf.random.set_seed(123)
    np.random.seed(123)
