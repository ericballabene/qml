import logging
from utils.helpers import setup_logging, setup_reproducibility
from data.loader import read_training_samples_limited, split_AB
from data.preprocessor import compute_feature_scalers, preprocess_data_dnn
from models.dnn_model import train_dnn_model
from inference.predictor import test_and_save
from config.settings import saved_model_A_DNN, saved_model_B_DNN

logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    setup_logging()
    setup_reproducibility()

    logger.info("Reading and preparing training data for DNN...")
    df = read_training_samples_limited(max_per_signal=5000, max_background_total=5000)
    compute_feature_scalers(df)
    df = df.sample(frac=1.0, random_state=47).reset_index(drop=True)
    A, B = split_AB(df)

    logger.info("Training DNN on subset A")
    model_A = train_dnn_model(A, saved_model_A_DNN)

    logger.info("Training DNN on subset B")
    model_B = train_dnn_model(B, saved_model_B_DNN)

    logger.info("Running inference with both models in one pass")
    test_and_save(model_A, model_B, output_dir='output_dnn', preprocess_fn=preprocess_data_dnn)

    logger.info("All done!")

if __name__ == '__main__':
    main()
