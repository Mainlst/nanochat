"""
ベースモデル用の統合評価スクリプトです。

3 つの評価モードをサポートします (カンマ区切り):
  --eval core    : CORE metric (ICL タスクの正解率)
  --eval bpb     : train/val split の bits per byte
  --eval sample  : モデルからサンプルを生成

デフォルトでは 3 つすべてを実行します: --eval core,bpb,sample

実行例:

    # HuggingFace モデル (例: GPT-2 124M) を 8 GPU で評価
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # nanochat モデル (例: d24) を 8 GPU で評価
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # 単一 GPU で簡易評価
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace 読み込みユーティリティ

class ModelWrapper:
    """HuggingFace モデルに nanochat 互換インターフェイスを持たせる軽量ラッパーです。"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """HuggingFace モデルとトークナイザーを読み込みます。"""
    print0(f"HuggingFace モデルを読み込み中: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """HuggingFace トークナイザー用の token_bytes tensor を計算します。"""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE 評価

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """eval_bundle.zip を展開し、base ディレクトリに配置します。"""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"eval_bundle ディレクトリを {eval_bundle_dir} に配置しました")


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    CORE ベンチマークでベースモデルを評価します。
    results, centered_results, core_metric を含む dict を返します。
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # 必要なら評価 bundle をダウンロード
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # ランダムベースライン値を読み込む
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # 各タスクを評価
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"評価中: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # max_per_task 使用時の subsampling を一貫させるためにシャッフル
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"正解率: {accuracy:.4f} | centered: {centered_result:.4f} | 時間: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# メイン処理

def main():
    parser = argparse.ArgumentParser(description="ベースモデル評価")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='実行する評価をカンマ区切りで指定: core,bpb,sample (デフォルト: すべて)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace モデルのパス (例: openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='checkpoint ディレクトリを識別する nanochat model tag')
    parser.add_argument('--step', type=int, default=None, help='読み込むモデル step (デフォルト: 最新)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='CORE 各タスクの最大サンプル数 (-1 = すべて)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='BPB 評価で使うデバイスごとのバッチサイズ')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='BPB で各 split を評価するトークン数')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (空なら自動検出)')
    args = parser.parse_args()

    # 評価モードを解析
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"無効な eval mode です: {invalid}. 有効値: {valid_modes}")

    # 分散実行と精度のセットアップ
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # モデルとトークナイザーを読み込む
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"評価対象モデル: {model_name}")
    print0(f"評価モード: {', '.join(sorted(eval_modes))}")

    # ログに残す結果
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- サンプリング ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("モデルサンプル")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\n条件付きサンプル:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\n無条件サンプル:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nHuggingFace モデルのサンプリングは未対応のためスキップします")

    # --- BPB 評価 ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB 評価")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # 最も近い倍数に調整
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"split_tokens を {args.split_tokens} に調整しました ({tokens_per_step} で割り切れる必要があります)")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE 評価 ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE 評価")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)

        # CSV 出力を書き込む
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\n結果を書き込みました: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- report にログ ---
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="ベースモデル評価", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
