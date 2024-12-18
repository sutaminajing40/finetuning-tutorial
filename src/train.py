import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from config import TRAIN_CONFIG
from data.dataset_loader import DatasetLoader
from models.model_factory import ModelFactory
from utils.logger import Logger
from utils.types import MetricsResult


def calculate_metrics(predictions: list[int], labels: list[int]) -> MetricsResult:
    """評価指標を計算する

    Args:
        predictions: 予測ラベルのリスト
        labels: 正解ラベルのリスト

    Returns:
        metrics: 評価指標
    """
    return MetricsResult(
        accuracy=accuracy_score(labels, predictions),
        f1_score=f1_score(labels, predictions, average="weighted"),
        precision=precision_score(labels, predictions, average="weighted"),
        recall=recall_score(labels, predictions, average="weighted"),
    )


def evaluate(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> MetricsResult:
    """モデルの評価を行う

    Args:
        model: 評価するモデル
        data_loader: データローダー

    Returns:
        metrics: 評価指標
    """
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(TRAIN_CONFIG.device) for k, v in batch.items()}
            outputs = model(batch)

            pred = outputs.logits.argmax(dim=-1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    return calculate_metrics(predictions, labels)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    logger: Logger,
) -> tuple[float, MetricsResult]:
    """1エポックの学習を行う

    Args:
        model: 学習するモデル
        optimizer: オプティマイザ
        train_loader: 学習用データローダー
        logger: ロガー

    Returns:
        avg_loss: 平均損失値
        metrics: 評価指標
    """
    model.train()
    total_loss = 0
    predictions = []
    labels = []

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        batch = {k: v.to(TRAIN_CONFIG.device) for k, v in batch.items()}

        outputs = model(batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = outputs.logits.argmax(dim=-1)
        predictions.extend(pred.cpu().numpy())
        labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(predictions, labels)

    return avg_loss, metrics


def main():
    """メイン関数"""
    logger = Logger()

    try:
        # データセットの読み込み
        logger.logger.info("Loading dataset...")
        dataset_loader = DatasetLoader()
        train_loader, val_loader, test_loader = dataset_loader.load_and_split()

        # モデルの作成
        logger.logger.info("Creating model...")
        model = ModelFactory.create_model()
        optimizer = ModelFactory.create_optimizer(model)

        # 学習前の評価
        logger.logger.info("Evaluating model before training...")
        initial_test_metrics = evaluate(model, test_loader)
        logger.logger.info("=== Initial Model Performance ===")
        logger.log_test_info(initial_test_metrics)

        # 学習ループ
        logger.logger.info("Starting training...")
        for epoch in range(TRAIN_CONFIG.epochs):
            # 学習
            loss, train_metrics = train_epoch(model, optimizer, train_loader, logger)
            logger.log_training_info(epoch + 1, loss, train_metrics)

            # 検証
            val_metrics = evaluate(model, val_loader)
            logger.log_validation_info(val_metrics)

        # 学習後のテスト
        logger.logger.info("Evaluating model after training...")
        final_test_metrics = evaluate(model, test_loader)
        logger.logger.info("=== Final Model Performance ===")
        logger.log_test_info(final_test_metrics)

        # 性能改善の表示
        logger.logger.info("=== Performance Improvement ===")
        accuracy_improvement = final_test_metrics.accuracy - initial_test_metrics.accuracy
        f1_improvement = final_test_metrics.f1_score - initial_test_metrics.f1_score
        logger.logger.info(f"Accuracy improvement: {accuracy_improvement:.4f}")
        logger.logger.info(f"F1-score improvement: {f1_improvement:.4f}")

        # モデルの保存
        torch.save(model.state_dict(), "models/sentiment_model.pth")
        logger.logger.info("Model saved successfully!")

    except Exception as e:
        logger.log_error(e)
        raise


if __name__ == "__main__":
    main()
