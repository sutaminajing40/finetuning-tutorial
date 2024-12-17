from typing import Dict, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import TRAIN_CONFIG


class SentimentDataset(Dataset):
    """感情分析用のデータセットクラス"""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer):
        """
        Args:
            texts: テキストのリスト
            labels: ラベルのリスト
            tokenizer: トークナイザー
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=TRAIN_CONFIG.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DatasetLoader:
    """データセットの読み込みと分割を行うクラス"""

    def __init__(self, dataset_name: str = "tyqiangz/multilingual-sentiments"):
        """
        Args:
            dataset_name: Hugging Faceのデータセット名
        """
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(TRAIN_CONFIG.model_name)

    def load_and_split(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """データセットを読み込み、学習/検証/テストに分割する

        Returns:
            train_loader: 学習用DataLoader
            val_loader: 検証用DataLoader
            test_loader: テスト用DataLoader
        """
        # Hugging Faceからデータセットを読み込む
        dataset = load_dataset(self.dataset_name, "japanese")

        # データセットの分割
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        # SentimentDatasetの作成
        train_data = SentimentDataset(
            texts=train_dataset["text"],
            labels=train_dataset["label"],
            tokenizer=self.tokenizer,
        )
        val_data = SentimentDataset(
            texts=validation_dataset["text"],
            labels=validation_dataset["label"],
            tokenizer=self.tokenizer,
        )
        test_data = SentimentDataset(
            texts=test_dataset["text"],
            labels=test_dataset["label"],
            tokenizer=self.tokenizer,
        )

        # DataLoaderの作成
        train_loader = DataLoader(
            train_data, batch_size=TRAIN_CONFIG.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=TRAIN_CONFIG.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_data, batch_size=TRAIN_CONFIG.batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader
