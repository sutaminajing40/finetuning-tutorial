import logging
from typing import Any

from utils.types import MetricsResult


class Logger:
    """ログ出力を行うクラス"""

    def __init__(self, name: str = "sentiment_analysis"):
        """
        Args:
            name: ロガーの名前
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # ハンドラーの設定
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # フォーマッターの設定
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def log_training_info(self, epoch: int, loss: float, metrics: MetricsResult) -> None:
        """学習情報をログ出力する

        Args:
            epoch: 現在のエポック数
            loss: 損失値
            metrics: 評価指標
        """
        self.logger.info(
            f"Epoch: {epoch} - "
            f"Loss: {loss:.4f} - "
            f"Accuracy: {metrics.accuracy:.4f} - "
            f"F1: {metrics.f1_score:.4f}"
        )

    def log_validation_info(self, metrics: MetricsResult) -> None:
        """検証情報をログ出力する

        Args:
            metrics: 評価指標
        """
        self.logger.info(
            "=== Validation Results ===\n"
            f"Accuracy: {metrics.accuracy:.4f}\n"
            f"F1-score: {metrics.f1_score:.4f}\n"
            f"Precision: {metrics.precision:.4f}\n"
            f"Recall: {metrics.recall:.4f}"
        )

    def log_test_info(self, metrics: MetricsResult) -> None:
        """テスト情報をログ出力する

        Args:
            metrics: 評価指標
        """
        self.logger.info(
            "=== Test Results ===\n"
            f"Accuracy: {metrics.accuracy:.4f}\n"
            f"F1-score: {metrics.f1_score:.4f}\n"
            f"Precision: {metrics.precision:.4f}\n"
            f"Recall: {metrics.recall:.4f}"
        )

    def log_error(self, error: Any) -> None:
        """エラー情報をログ出力する

        Args:
            error: エラー情報
        """
        self.logger.error(f"Error occurred: {str(error)}")
