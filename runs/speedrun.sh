#!/bin/bash

# このスクリプトは独自のGPT-2級大規模言語モデル（事前学習＋ファインチューニング）を学習するよう構成されています。
# 何も入っていない8基のH100 GPUノードで動作することを想定しており、完了まで約3時間かかります。

# 1) 実行例（最も簡単な方法）
# bash runs/speedrun.sh
#
# 2) screenセッション内での実行例（約3時間かかるため推奨）
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
#
# 3) Weights & Biases (wandb) によるログ記録を有効にする例
# （事前にwandbの設定が必要）
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# 中間生成物の保存先
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# uv を使った Python 仮想環境のセットアップ

# uv が未インストールなら導入
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# ローカル仮想環境を作成（未作成の場合）
[ -d ".venv" ] || uv venv

# リポジトリの依存関係をインストール
uv sync --extra gpu

# 仮想環境を有効化
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb の設定
#
# wandb を利用する場合:
#
# 1. 事前にログイン
#    wandb login
#
# 2. 実行時に WANDB_RUN を指定
#    WANDB_RUN=d26 bash speedrun.sh
#
if [ -z "$WANDB_RUN" ]; then
    # 指定がなければ "dummy" を使用
    # 特別扱いされ、wandbへのログ送信を行わない
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 実行レポートの初期化
#
# report/ ディレクトリを初期化し、
# システム情報や開始時刻を含むヘッダーを書き込む
#
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# トークナイザー学習

# 事前学習データセットの先頭約20億文字を取得
#
# 各シャードは約2.5億文字
# 20億 / 2.5億 = 8 シャード
#
# 各シャードは圧縮後約100MB
# 合計で約800MB程度
#
# データ生成方法は
# dev/repackage_data_reference.py を参照
#
python -m nanochat.dataset -n 8

# トークナイザー学習中に追加データをバックグラウンドで取得
#
# GPT-2級性能には約150シャード必要
# 余裕を見て170シャード取得
#
# 利用可能な総シャード数は6542
#
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# 語彙数 32768 (= 2^15) のトークナイザーを学習
python -m scripts.tok_train

# トークナイザー評価
# （圧縮率などを測定）
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基盤モデル（事前学習）

echo "データセットのダウンロード完了を待機中..."
wait $DATASET_DOWNLOAD_PID

# d24 モデル
#
# GPT-2を上回る性能を狙うため、
# 計算量最適比率10.5から8へ下げて
# やや学習不足気味に設定
#
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN

# モデル評価
#
# - CORE指標
# - 学習/検証データのBPB
# - サンプル生成
#
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_eval -- \
    --device-batch-size=16

# -----------------------------------------------------------------------------
# 教師ありファインチューニング（SFT）
#
# 会話用特殊トークン
# ツール利用
# 選択式応答
#
# などを学習

# 人格付与用の合成会話データ（約2.3MB）を取得
#
# データ生成方法は
# dev/gen_synthetic_data.py を参照
#
curl -L \
    -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# SFT学習
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN

# SFT後の評価
torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_eval -- \
    -i sft

# -----------------------------------------------------------------------------
# CLIで対話
#
# -p を省略すると対話モード
#
# python -m scripts.chat_cli -p "空はなぜ青いのですか？"

# Web UIで対話
#
# ChatGPT風インターフェース
#
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 最終レポート生成
#
# report/ 内の各セクションを統合し
# report.md を生成
#
# 利便性のためカレントディレクトリにもコピーされる
#
python -m nanochat.report generate