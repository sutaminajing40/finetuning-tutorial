import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from config import TRAIN_CONFIG
from utils.types import BatchType, ModelOutput


class SentimentClassifier(nn.Module):
    """感情分析用のBERTモデルクラス"""

    def __init__(self, num_labels: int = 3):
        """
        Args:
            num_labels: 分類するラベルの数（positive, neutral, negative）
        """
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            TRAIN_CONFIG.model_name, num_labels=num_labels
        )

    def forward(self, batch: BatchType) -> ModelOutput:
        """順伝播の計算を行う

        Args:
            batch: 入力バッチ（input_ids, attention_mask, labels）

        Returns:
            ModelOutput: モデルの出力（logits, loss）
        """
        outputs = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels", None),
        )

        return ModelOutput(logits=outputs.logits, loss=outputs.loss)


class ModelFactory:
    """モデルの作成と設定を行うクラス"""

    @staticmethod
    def create_model() -> SentimentClassifier:
        """モデルを作成し、デバイスに移動する

        Returns:
            model: 作成したモデル
        """
        model = SentimentClassifier()
        model.to(TRAIN_CONFIG.device)
        return model

    @staticmethod
    def create_optimizer(model: nn.Module) -> torch.optim.Optimizer:
        """オプティマイザを作成する

        Args:
            model: 最適化対象のモデル

        Returns:
            optimizer: 作成したオプティマイザ
        """
        return torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG.learning_rate)
