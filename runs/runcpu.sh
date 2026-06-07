#!/bin/bash

# CPU（またはMacbookの場合はMPS）上でコードパスの一部をテストするための実行例
# このスクリプトは2026年1月17日に最終更新・最適化されました

# 実行方法:
# bash runs/runcpu.sh

# 注意: LLMのトレーニングにはGPU計算リソースと費用が必要です。Macbookでは十分な性能は期待できません。
# この実行例は教育的・デモンストレーション用のものであり、実用的な用途には適さないことをご了承ください。
# 必要に応じて、このスクリプトを手動で1つずつ実行し、コマンドをターミナルにコピー＆ペーストして使用することも可能です。

# 必要な環境設定
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
# 約20億文字でトークナイザーをトレーニング（私のMacBook Pro M3 Maxで約34秒）
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# 小規模な4層モデルをトレーニング
# この設定は私のMacBook Pro M3 Maxで約30分で完了するように調整しています
# より良い結果を得るには、num_iterationsを増やすか、他のLLMから得た知見を参考にしてください
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16
# SFTトレーニング（私のMacBook Pro M3 Maxで約10分）
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# CLI経由でモデルと対話
# モデルは「私はパリです」といった自己紹介ができるはずです
# 空の色が青であることまで知っているかもしれません
# 質問する前に「こんにちは」と挨拶すると、モデルがより適切に反応する場合があります
# python -m scripts.chat_cli -p "フランスの首都はどこですか？"

# 美しいWebUIインターフェース（ChatGPT風）でモデルと対話
# python -m scripts.chat_web
