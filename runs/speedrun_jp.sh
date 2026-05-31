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

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# llm-jp-corpus v3 を nanochat 用 parquet shard へ変換
#
# nanochat の既存 dataloader は $NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet の
# text カラムを読むため、llm-jp の jsonl.gz を同じ形式へ変換します。

python - <<'PY'
import gzip
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

base_dir = Path(os.environ["NANOCHAT_BASE_DIR"])
repo_dir = base_dir / "llm-jp-corpus-v3"
raw_dir = base_dir / "llmjp_raw"
data_dir = base_dir / "base_data_climbmix"
manifest_path = data_dir / "llmjp_manifest.txt"

repo_url = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3.git"
raw_base_url = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main"
include_prefixes = [p.strip() for p in os.environ["LLMJP_INCLUDE_PREFIXES"].split(",") if p.strip()]
num_train_files = int(os.environ["LLMJP_NUM_TRAIN_FILES"])
target_chars = int(os.environ["LLMJP_SHARD_CHARS"])
rebuild = os.environ["LLMJP_REBUILD_DATA"] == "1"

if target_chars <= 0:
    raise SystemExit("LLMJP_SHARD_CHARS must be positive")

def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kwargs)

def ensure_metadata_repo():
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], env=env)
    else:
        run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", "main"], env=env)
        run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"], env=env)

def git_files():
    out = subprocess.check_output(["git", "-C", str(repo_dir), "ls-files"], text=True)
    return sorted(p for p in out.splitlines() if p.endswith(".jsonl.gz"))

def included(path):
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in include_prefixes)

def prefix_rank(path):
    for i, prefix in enumerate(include_prefixes):
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return i
    return len(include_prefixes)

def is_validation(path):
    name = Path(path).name
    return "validation" in name or "eval" in name

def choose_files(paths):
    candidates = sorted((p for p in paths if included(p)), key=lambda p: (prefix_rank(p), p))
    val = [p for p in candidates if is_validation(p)]
    train = [p for p in candidates if not is_validation(p)]
    if not train:
        raise SystemExit(f"No train files matched LLMJP_INCLUDE_PREFIXES={include_prefixes}")
    if not val:
        raise SystemExit(f"No validation/eval file matched LLMJP_INCLUDE_PREFIXES={include_prefixes}")
    if num_train_files > 0:
        train = train[:num_train_files]
    # nanochat は最後の parquet を validation として扱うので、validation は最後に置く。
    return train, [val[0]]

def current_manifest(train_paths, val_paths):
    lines = [
        "llm-jp-corpus-v3",
        "include_prefixes=" + ",".join(include_prefixes),
        f"num_train_files={num_train_files}",
        f"target_chars={target_chars}",
        "train:",
        *train_paths,
        "validation:",
        *val_paths,
    ]
    return "\n".join(lines) + "\n"

def download(path):
    dst = raw_dir / path
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    url = raw_base_url + "/" + urllib.parse.quote(path)
    print(f"Downloading {path}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    tmp.replace(dst)
    return dst

def iter_texts(gz_path):
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if isinstance(text, str) and text:
                yield text

def write_shard(texts, shard_index):
    out = data_dir / f"shard_{shard_index:05d}.parquet"
    table = pa.Table.from_pydict({"text": texts})
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        row_group_size=1024,
        write_statistics=False,
    )
    tmp.replace(out)
    print(f"Wrote {out} ({len(texts):,} docs)", flush=True)

def convert(paths, shard_start):
    shard_index = shard_start
    texts = []
    chars = 0
    for path in paths:
        gz_path = download(path)
        for text in iter_texts(gz_path):
            texts.append(text)
            chars += len(text)
            if chars >= target_chars:
                write_shard(texts, shard_index)
                shard_index += 1
                texts = []
                chars = 0
    if texts:
        write_shard(texts, shard_index)
        shard_index += 1
    return shard_index

ensure_metadata_repo()
all_paths = git_files()
train_paths, val_paths = choose_files(all_paths)
manifest = current_manifest(train_paths, val_paths)

if rebuild and data_dir.exists():
    shutil.rmtree(data_dir)
data_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)

existing_parquets = sorted(data_dir.glob("shard_*.parquet"))
if existing_parquets and manifest_path.exists() and manifest_path.read_text() == manifest:
    print(f"Using existing llm-jp parquet shards in {data_dir}", flush=True)
    sys.exit(0)

for path in data_dir.glob("shard_*.parquet"):
    path.unlink()
for path in data_dir.glob("*.tmp"):
    path.unlink()

print(f"Selected {len(train_paths)} train files and {len(val_paths)} validation file from llm-jp-corpus v3", flush=True)
print("Include prefixes: " + ", ".join(include_prefixes), flush=True)

next_shard = convert(train_paths, 0)
if next_shard == 0:
    raise SystemExit("No train parquet shard was written")
convert(val_paths, next_shard)
manifest_path.write_text(manifest)
print(f"Done. Wrote llm-jp parquet shards to {data_dir}", flush=True)
PY

# -----------------------------------------------------------------------------
# トークナイザー学習

python -m scripts.tok_train \
    --max-chars="$TOKENIZER_MAX_CHARS" \
    --doc-cap="$TOKENIZER_DOC_CAP" \
    --vocab-size="$TOKENIZER_VOCAB_SIZE"
if [ "$RUN_TOK_EVAL" = "1" ]; then
    python -m scripts.tok_eval
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
#   python -m scripts.base_eval --device-batch-size=1 --eval=sample
#
# SFT を有効にした場合:
#   python -m scripts.chat_cli -p "日本語で自己紹介してください。"

python -m nanochat.report generate
