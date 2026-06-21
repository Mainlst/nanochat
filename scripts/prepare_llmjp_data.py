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


BASE_DIR = Path(os.environ["NANOCHAT_BASE_DIR"])
REPO_DIR = BASE_DIR / "llm-jp-corpus-v3"
RAW_DIR = BASE_DIR / "llmjp_raw"
DATA_DIR = BASE_DIR / "base_data_climbmix"
MANIFEST_PATH = DATA_DIR / "llmjp_manifest.txt"

REPO_URL = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3.git"
RAW_BASE_URL = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main"


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_metadata_repo():
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    if not (REPO_DIR / ".git").exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], env=env)
    else:
        run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", "main"], env=env)
        run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/main"], env=env)


def git_files():
    out = subprocess.check_output(["git", "-C", str(REPO_DIR), "ls-files"], text=True)
    return sorted(p for p in out.splitlines() if p.endswith(".jsonl.gz"))


def included(path, include_prefixes):
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in include_prefixes)


def prefix_rank(path, include_prefixes):
    for i, prefix in enumerate(include_prefixes):
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            return i
    return len(include_prefixes)


def is_validation(path):
    name = Path(path).name
    return "validation" in name or "eval" in name


def choose_files(paths, include_prefixes, num_train_files):
    candidates = sorted(
        (p for p in paths if included(p, include_prefixes)),
        key=lambda p: (prefix_rank(p, include_prefixes), p),
    )
    val = [p for p in candidates if is_validation(p)]
    train = [p for p in candidates if not is_validation(p)]
    if not train:
        raise SystemExit(f"No train files matched LLMJP_INCLUDE_PREFIXES={include_prefixes}")
    if not val:
        raise SystemExit(f"No validation/eval file matched LLMJP_INCLUDE_PREFIXES={include_prefixes}")
    if num_train_files > 0:
        train = train[:num_train_files]
    # nanochat treats the last parquet file as validation, so write validation last.
    return train, [val[0]]


def current_manifest(train_paths, val_paths, include_prefixes, num_train_files, target_chars):
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
    dst = RAW_DIR / path
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    url = RAW_BASE_URL + "/" + urllib.parse.quote(path)
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
    out = DATA_DIR / f"shard_{shard_index:05d}.parquet"
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


def convert(paths, shard_start, target_chars):
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


def main():
    include_prefixes = [p.strip() for p in os.environ["LLMJP_INCLUDE_PREFIXES"].split(",") if p.strip()]
    num_train_files = int(os.environ["LLMJP_NUM_TRAIN_FILES"])
    target_chars = int(os.environ["LLMJP_SHARD_CHARS"])
    rebuild = os.environ["LLMJP_REBUILD_DATA"] == "1"

    if target_chars <= 0:
        raise SystemExit("LLMJP_SHARD_CHARS must be positive")

    ensure_metadata_repo()
    all_paths = git_files()
    train_paths, val_paths = choose_files(all_paths, include_prefixes, num_train_files)
    manifest = current_manifest(train_paths, val_paths, include_prefixes, num_train_files, target_chars)

    if rebuild and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_parquets = sorted(DATA_DIR.glob("shard_*.parquet"))
    if existing_parquets and MANIFEST_PATH.exists() and MANIFEST_PATH.read_text() == manifest:
        print(f"Using existing llm-jp parquet shards in {DATA_DIR}", flush=True)
        sys.exit(0)

    for path in DATA_DIR.glob("shard_*.parquet"):
        path.unlink()
    for path in DATA_DIR.glob("*.tmp"):
        path.unlink()

    print(
        f"Selected {len(train_paths)} train files and {len(val_paths)} validation file from llm-jp-corpus v3",
        flush=True,
    )
    print("Include prefixes: " + ", ".join(include_prefixes), flush=True)

    next_shard = convert(train_paths, 0, target_chars)
    if next_shard == 0:
        raise SystemExit("No train parquet shard was written")
    convert(val_paths, next_shard, target_chars)
    MANIFEST_PATH.write_text(manifest)
    print(f"Done. Wrote llm-jp parquet shards to {DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
