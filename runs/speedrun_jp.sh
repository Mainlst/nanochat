#!/bin/bash
set -euo pipefail

# llm-jp-corpus v3 の日本語コーパスで nanochat の基盤モデルを事前学習するスクリプトです。
#
# 実行例:
#   bash runs/speedrun_jp.sh
#
# データ量やモデルサイズを調整する例:
#   LLMJP_NUM_TRAIN_FILES=20 DEPTH=16 NPROC_PER_NODE=4 bash runs/speedrun_jp.sh
#
# 注意:
# - llm-jp-corpus v3 は合計 2.7TB / 1.7T tokens 規模のコーパスです。
# - このスクリプトのデフォルトは日本語サブセットの一部を使う speedrun 設定です。
# - SFT は既存の英語中心タスクを使うため、デフォルトでは実行しません。

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_llmjp}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# 設定

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEPTH="${DEPTH:-24}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
# 空文字を指定した場合は FP8 を無効化できるようにする。
# 例: RTX 3070 では `FP8_FLAG=` を指定する。
FP8_FLAG="${FP8_FLAG---fp8}"
EVAL_TOKENS="${EVAL_TOKENS:-41943040}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-2000}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2000}"
BASE_EVAL="${BASE_EVAL:-1}"
BASE_EVAL_MODES="${BASE_EVAL_MODES:-core,bpb,sample}"
BASE_EVAL_SPLIT_TOKENS="${BASE_EVAL_SPLIT_TOKENS:-20971520}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"
TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
RUN_TOK_EVAL="${RUN_TOK_EVAL:-1}"

# llm-jp-corpus v3 の日本語サブセットから、先頭 N 個の train jsonl.gz を使います。
# 0 以下にすると対象サブセット内の train ファイルをすべて使います。
LLMJP_NUM_TRAIN_FILES="${LLMJP_NUM_TRAIN_FILES:-84}"

# 1 parquet shard あたりの目安文字数。大きくするとファイル数は減りますが変換時メモリを使います。
LLMJP_SHARD_CHARS="${LLMJP_SHARD_CHARS:-250000000}"

# デフォルトでは、比較的ライセンスと品質を追いやすい日本語サブセットを選びます。
# 全日本語データを使いたい場合は LLMJP_INCLUDE_PREFIXES="ja/" を指定してください。
LLMJP_INCLUDE_PREFIXES="${LLMJP_INCLUDE_PREFIXES:-ja/ja_wiki,ja/kaken,ja/ja_warp_html/level0,ja/ja_cc/level0}"

# 既存の変換済み parquet を捨てて作り直す場合は 1。
LLMJP_REBUILD_DATA="${LLMJP_REBUILD_DATA:-0}"

export LLMJP_NUM_TRAIN_FILES
export LLMJP_SHARD_CHARS
export LLMJP_INCLUDE_PREFIXES
export LLMJP_REBUILD_DATA

# SFT は既存の SmolTalk/MMLU/GSM8K 等で英語寄りになるため、必要な場合だけ有効化します。
RUN_SFT="${RUN_SFT:-0}"

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# uv を使った Python 仮想環境のセットアップ

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 実行レポートの初期化

python3 -m nanochat.report reset

# -----------------------------------------------------------------------------
# llm-jp-corpus v3 を nanochat 用 parquet shard へ変換
#
# nanochat の既存 dataloader は $NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet の
# text カラムを読むため、llm-jp の jsonl.gz を同じ形式へ変換します。

python3 -m scripts.prepare_llmjp_data

# -----------------------------------------------------------------------------
# トークナイザー学習

python3 -m scripts.tok_train \
    --max-chars="$TOKENIZER_MAX_CHARS" \
    --doc-cap="$TOKENIZER_DOC_CAP" \
    --vocab-size="$TOKENIZER_VOCAB_SIZE"
if [ "$RUN_TOK_EVAL" = "1" ]; then
    python3 -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# 基盤モデル（事前学習）

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m scripts.base_train -- \
    --depth="$DEPTH" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --num-iterations="$NUM_ITERATIONS" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every="$CORE_METRIC_EVERY" \
    --sample-every="$SAMPLE_EVERY" \
    $FP8_FLAG \
    --run="$WANDB_RUN"

# 評価。CORE は英語中心の評価を含みますが、train/val BPB の確認にも使います。
if [ "$BASE_EVAL" = "1" ]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m scripts.base_eval -- \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --eval="$BASE_EVAL_MODES" \
        --split-tokens="$BASE_EVAL_SPLIT_TOKENS"
fi

# -----------------------------------------------------------------------------
# 任意: SFT

if [ "$RUN_SFT" = "1" ]; then
    curl -L \
        -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m scripts.chat_sft -- \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --run="$WANDB_RUN"

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
        -m scripts.chat_eval -- \
        -i sft
fi

# -----------------------------------------------------------------------------
# 生成例
#
# 基盤モデルだけ試す:
#   python3 -m scripts.base_eval --device-batch-size=1 --eval=sample
#
# SFT を有効にした場合:
#   python3 -m scripts.chat_cli -p "日本語で自己紹介してください。"

python3 -m nanochat.report generate
