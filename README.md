# BERT を用いた感情分析ファインチューニング

## プロジェクト概要

このプロジェクトは、bert-base-japanese-v3 モデルを使用して感情分析（ポジティブ/ネガティブ）のファインチューニングを行うものです。

## 使用モデル

- モデル名: bert-base-japanese-v3
- 開発元: 東北大学
- 特徴: 日本語に特化、SentencePiece によるトークナイズ

## 環境設定

1. 必要なパッケージのインストール

```bash
pip install -r requirements.txt
```

2. データセットの準備

- chABSA-dataset をダウンロードし、`data`ディレクトリに配置してください。

## 使用方法

1. モデルの学習

```bash
python src/train.py
```

2. モデルの評価

```bash bash
python src/evaluate.py
```

## プロジェクト構造

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data/
│   │   └── dataset_loader.py
│   ├── models/
│   │   └── model_factory.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── types.py
│       └── logger.py
└── tests/
    └── test_dataset_loader.py
```

## 主要パラメータ

- バッチサイズ: 16
- エポック数: 3-5
- 学習率: 2e-5
- 最大シーケンス長: 128

## 評価指標

- Accuracy
- F1-score
