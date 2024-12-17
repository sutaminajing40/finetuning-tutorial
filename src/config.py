import torch

from utils.types import TrainConfig

# モデル設定
MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
MAX_LENGTH = 128

# 学習設定
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# データセット設定
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 学習設定をTrainConfigとして定義
TRAIN_CONFIG = TrainConfig(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    max_length=MAX_LENGTH,
    model_name=MODEL_NAME,
    device=DEVICE,
)

# ラベル定義
LABEL_MAP = {
    "positive": 0,
    "neutral": 1,
    "negative": 2,
}

# 乱数シード
RANDOM_SEED = 42
