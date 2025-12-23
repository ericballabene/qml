import logging
from utils.helpers import setup_logging, setup_reproducibility
from data.loader import read_training_samples_limited, split_AB
from data.preprocessor import compute_feature_scalers
from models.quantum_model import train_qnn_model
from inference.predictor import test_and_save
from config.settings import saved_model_A, saved_model_B

logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    setup_logging()
    setup_reproducibility()

    logger.info("Reading and preparing training data for QNN...")
    df = read_training_samples_limited(max_per_signal=5000, max_background_total=5000)
    compute_feature_scalers(df)
    A, B = split_AB(df)

    logger.info("Training QNN on subset A")
    model_A = train_qnn_model(A, saved_model_A)

    logger.info("Training QNN on subset B")
    model_B = train_qnn_model(B, saved_model_B)

    logger.info("Running inference with both models in one pass")
    test_and_save(model_A, model_B)

    logger.info("All done!")

if __name__ == '__main__':
    main()
