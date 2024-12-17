from typing import Any, Dict, NamedTuple, Union

from torch import Tensor

# データセット関連の型定義
DatasetDict = Dict[str, Any]
BatchType = Dict[str, Tensor]


class TrainConfig(NamedTuple):
    """学習設定を保持するクラス"""

    batch_size: int
    epochs: int
    learning_rate: float
    max_length: int
    model_name: str
    device: str


class ModelOutput(NamedTuple):
    """モデルの出力を保持するクラス"""

    logits: Tensor
    loss: Union[Tensor, None]


class MetricsResult(NamedTuple):
    """評価指標の結果を保持するクラス"""

    accuracy: float
    f1_score: float
    precision: float
    recall: float
