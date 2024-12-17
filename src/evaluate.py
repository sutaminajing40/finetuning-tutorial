import torch

from data.dataset_loader import DatasetLoader
from models.model_factory import ModelFactory
from train import evaluate
from utils.logger import Logger


def main():
    """メイン関数"""
    logger = Logger()

    try:
        # データセットの読み込み
        logger.logger.info("Loading dataset...")
        dataset_loader = DatasetLoader()
        _, _, test_loader = dataset_loader.load_and_split()

        # モデルの読み込み
        logger.logger.info("Loading model...")
        model = ModelFactory.create_model()
        model.load_state_dict(torch.load("models/sentiment_model.pth"))

        # テスト
        logger.logger.info("Evaluating model...")
        test_metrics = evaluate(model, test_loader)
        logger.log_test_info(test_metrics)

    except Exception as e:
        logger.log_error(e)
        raise


if __name__ == "__main__":
    main()
