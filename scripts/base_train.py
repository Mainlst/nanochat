"""
モデルを学習します。プロジェクトのルートディレクトリから次のように実行します:

python -m scripts.base_train

分散実行する場合:

torchrun --nproc_per_node=8 -m scripts.base_train

CPU/Macbook だけで実行する場合は、かなり小さい LLM を学習する設定にしてください。例:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import argparse
import gc
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
)
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3, USE_FA3
from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_core

print_banner()

SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


# -----------------------------------------------------------------------------
# CLI 引数


def build_arg_parser():
    parser = argparse.ArgumentParser(description="ベースモデルを事前学習します", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="このヘルプを表示して終了")

    # ログ
    parser.add_argument("--run", type=str, default="dummy", help="wandb の run 名 ('dummy' で wandb ログを無効化)")

    # 実行環境
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (空なら自動検出)")

    # FP8 学習
    parser.add_argument("--fp8", action="store_true", help="FP8 学習を有効化 (H100 以降の GPU と torchao が必要)")
    parser.add_argument(
        "--fp8-recipe",
        type=str,
        default="tensorwise",
        choices=["rowwise", "tensorwise"],
        help="FP8 スケーリング方式: tensorwise (高速、推奨) または rowwise (高精度だが低速)",
    )

    # モデル構造
    parser.add_argument("--depth", type=int, default=20, help="Transformer モデルの深さ")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=128, help="attention の目標 head 次元")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="最大コンテキスト長")
    parser.add_argument(
        "--window-pattern",
        type=str,
        default="SSSL",
        help="各層に繰り返し適用する sliding window パターン: L=full, S=half context (例: 'SSL')",
    )

    # 学習期間 (優先順位に従って 1 つだけ使用)
    parser.add_argument("--num-iterations", type=int, default=-1, help="最適化ステップ数を明示指定 (-1 で無効)")
    parser.add_argument("--target-flops", type=float, default=-1.0, help="target_flops に到達する num_iterations を計算 (-1 で無効)")
    parser.add_argument(
        "--target-param-data-ratio",
        type=float,
        default=12,
        help="data:param 比率を維持する num_iterations を計算 (Chinchilla=20, -1 で無効)",
    )

    # 最適化
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=32,
        help="デバイスごとのバッチサイズ。VRAM が OOM する場合は 16,8,4,... に下げるとよいです。",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=-1,
        help="トークン単位の合計バッチサイズ。例: 524288。(-1 で最適値を自動計算)",
    )
    parser.add_argument("--embedding-lr", type=float, default=0.3, help="embedding パラメータの学習率 (Adam)")
    parser.add_argument("--unembedding-lr", type=float, default=0.008, help="unembedding パラメータの学習率 (Adam)")
    parser.add_argument("--weight-decay", type=float, default=0.28, help="Muon optimizer 用の控えめな weight decay (weights 用)")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="行列パラメータの学習率 (Muon)")
    parser.add_argument("--scalar-lr", type=float, default=0.5, help="スカラー (resid_lambdas, x0_lambdas) の学習率")
    parser.add_argument("--warmup-steps", type=int, default=40, help="LR warmup のステップ数")
    parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="LR warmdown に使う iteration の比率")
    parser.add_argument("--final-lr-frac", type=float, default=0.05, help="初期 LR に対する最終 LR の比率")
    parser.add_argument("--resume-from-step", type=int, default=-1, help="このステップから学習を再開 (-1 で無効)")

    # 評価
    parser.add_argument("--eval-every", type=int, default=250, help="N ステップごとに val bpb を評価 (-1 で無効)")
    parser.add_argument("--eval-tokens", type=int, default=80 * 524288, help="val loss 評価に使うトークン数")
    parser.add_argument("--core-metric-every", type=int, default=2000, help="N ステップごとに CORE metric を評価 (-1 で無効)")
    parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="CORE metric の各タスクで使う最大サンプル数")
    parser.add_argument("--sample-every", type=int, default=2000, help="N ステップごとにモデルからサンプル生成 (-1 で無効)")
    parser.add_argument("--save-every", type=int, default=-1, help="N ステップごとに checkpoint を保存 (-1 なら最後のみ)")

    # 出力
    parser.add_argument("--model-tag", type=str, default=None, help="checkpoint ディレクトリ名に使う model tag を上書き")
    return parser


args = build_arg_parser().parse_args()
user_config = vars(args).copy()  # ログ用
# -----------------------------------------------------------------------------
# compute 初期化と wandb ログ

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # このプロセスがログ出力や checkpoint 保存などを担当
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | ピーク FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # CPU/MPS では MFU に意味がない
print0(f"計算 dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb ログ初期化
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=args.run, config=user_config)
)

# Flash Attention の状態
if USE_FA3:
    print0("✓ Flash Attention 3 を使用します (Hopper GPU を検出)。効率的で新しい実装です。")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(
            f"警告: Flash Attention 3 は bf16 のみ対応ですが、COMPUTE_DTYPE={COMPUTE_DTYPE} です。"
            "PyTorch SDPA fallback を使用します"
        )
    else:
        print0("警告: Flash Attention 3 が利用できないため、PyTorch SDPA fallback を使用します")
    print0("警告: FA3 がないため学習効率は下がります")
    if args.window_pattern != "L":
        print0(
            f"警告: SDPA は sliding window attention に対応していません "
            f"(window_pattern='{args.window_pattern}')。GPU 使用率が大きく低下します。"
        )
        print0(
            "警告: sliding window pattern を交互に使わず、full context attention にするため "
            "--window-pattern L の使用を推奨します。"
        )
    print0("!" * 80)

# -----------------------------------------------------------------------------
# tokenizer は評価に使い、モデル初期化には vocab size も必要
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"語彙サイズ: {vocab_size:,}")

# -----------------------------------------------------------------------------
# モデル初期化

def build_model_meta(depth):
    """指定した depth のモデルを meta device 上に構築します (shape/dtype のみで実データなし)。"""
    # model_dim はきれいに割り切れるよう head_dim の最寄りの倍数へ切り上げる
    # (FA3 は head_dim が 8 で割り切れる必要があり、これで head_dim == args.head_dim も保証される)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


# モデルを構築し、デバイスへ移動して重みを初期化
model = build_model_meta(args.depth)  # 1) meta デバイス上に構築 (shape/dtype のみで実データなし)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"モデル設定:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)  # 2) すべての tensor に対象デバイス上の storage を割り当てる (値は未初期化)
model.init_weights()  # 3) すべての tensor を初期化

# 再開する場合は checkpoint のモデルパラメータで上書き
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"step {args.resume_from_step} から最適化を再開します")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # コピー後にメモリを解放

# -----------------------------------------------------------------------------
# FP8 学習の初期化と管理 (torch.compile より前に行う必要がある)


def is_fp8_eligible_linear(mod):
    if not isinstance(mod, torch.nn.Linear):
        return False
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    if min(mod.in_features, mod.out_features) < 128:
        return False
    return True


def fp8_module_filter(mod, _fqn):
    # FP8 ハードウェア要件として次元が 16 で割り切れ、かつ十分大きいものだけを対象にする。
    return is_fp8_eligible_linear(mod)


# --fp8 が指定された場合は Linear layer を Float8Linear に変換
if args.fp8:
    if device_type != "cuda":
        print0("警告: FP8 学習には CUDA が必要です。--fp8 フラグを無視します")
    else:
        # 独自 fp8 は torchao より単純だが、API 互換になるよう実装している
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(
            f"✓ FP8 学習を有効化しました ({args.fp8_recipe} scaling) - "
            f"{num_fp8}/{num_linear} 個の linear layer を変換し、{num_skipped} 個をスキップしました (小さすぎるため)"
        )


def iter_fp8_modules(model):
    """(親 module, 属性名, Float8Linear module) の組を yield します。"""
    for name, module in model.named_modules():
        if "Float8" not in type(module).__name__:
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name
        yield parent, attr_name, module


def linear_from_fp8(fp8_module):
    """FP8 module のパラメータを共有する nn.Linear 互換の view を作ります。"""
    linear = Linear(
        fp8_module.in_features,
        fp8_module.out_features,
        bias=fp8_module.bias is not None,
        device="meta",  # 不要な VRAM 確保を避けるため meta デバイスを使う。
        dtype=fp8_module.weight.dtype,
    )
    linear.weight = fp8_module.weight  # コピーせず共有
    if fp8_module.bias is not None:
        linear.bias = fp8_module.bias
    return linear

# モデル評価を BF16 のまま行うため、FP8 を一時的に無効化する context manager
@contextmanager
def disable_fp8(model):
    """BF16 評価のために Float8Linear module を一時的に nn.Linear と差し替えます。

    CastConfig は frozen dataclass なので scaling_type を変更できません。
    代わりに Float8Linear module 自体を差し替え、終了後に復元します。
    """
    fp8_locations = list(iter_fp8_modules(model))

    if not fp8_locations:
        yield  # FP8 module がなければ何もしない
        return

    # Float8Linear -> Linear に差し替える (入力 dtype に合わせて weight を cast する独自 class)
    # VRAM 使用量の急増を避けるため device="meta" を使い、weight tensor は後から差し替える
    for parent, attr_name, fp8_module in fp8_locations:
        setattr(parent, attr_name, linear_from_fp8(fp8_module))

    try:
        yield
    finally:
        # Float8Linear module を復元
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# モデルを compile

orig_model = model  # 未 compile の元モデル。生の state_dict 保存や推論/評価に使う
model = torch.compile(model, dynamic=False)  # 入力 shape は変わらないので dynamic=False でよい

# -----------------------------------------------------------------------------
# scaling law と muP 外挿で、最適な学習期間・バッチサイズ・学習率・weight decay を決める

# モデルのパラメータ数を取得
param_counts = model.num_scaling_params()
print0("パラメータ数:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"1 トークンあたりの推定 FLOPs: {num_flops_per_token:e}")


# 1) scaling law を使い、token 単位の最適な学習期間を決める
# compute-optimal なモデルは --target-param-data-ratio の Tokens:Params 比を満たす
# (scaling law 解析から実験的に導出)。モデルは初期化済みなので Params は分かっている。
# 最適 token 数は単に target-param-data-ratio * Params になる。
def get_scaling_params(m):
    # どの params を使うかについては、transformer matrices + lm_head が最もきれいな scaling law になる
    # (dev/LOG.md 2026-01-27 参照)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    return scaling_params


num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)  # これから学習するモデルの最適 token 数

# 参照モデルは d12。多くの hyperparameter はここで調整し、より深いモデルへ転移する (muP 風)
d12_ref = build_model_meta(12)  # meta デバイス上にモデルを作る
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)  # d12 の compute-optimal な token 単位の学習期間 (実測)
B_REF = 2**19  # d12 における最適バッチサイズ ~= 524,288 tokens (実測)

# 2) token horizon が分かったので、最適なバッチサイズを計算できる
# Power Lines 論文に従う (Bopt ∝ D^0.383)。参照: https://arxiv.org/abs/2505.13738
# 最適バッチサイズは約 D^0.383 で増える。例: D が d12 から d24 へ倍増すると、B は 2^0.383 ≈ 1.3 倍になる。
total_batch_size = args.total_batch_size  # ユーザー指定による上書きが可能
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))  # 効率のため最寄りの 2 の冪に丸める
    print0(f"最適 batch size を自動計算しました: {total_batch_size:,} tokens")

# 3) バッチサイズが分かったので、学習率の補正を計算する (バッチサイズが大きいほど高い学習率にできる)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF  # B/B_ref
if batch_ratio != 1.0:
    # SGD: バッチサイズに対する linear scaling が標準 (nanochat では未使用)
    # AdamW: sqrt scaling が標準: η ∝ √(B/B_ref)
    # Muon: AdamW と同じ scaling を使う: η ∝ √(B/B_ref) (十分には検証していない仮定)
    batch_lr_scale = batch_ratio ** 0.5  # η ∝ √(B/B_ref)
    print0(f"batch size {total_batch_size:,} に合わせて LR を {batch_lr_scale:.4f} 倍します (基準: {B_REF:,})")

# 4) バッチサイズと token horizon が分かったので、適切な weight decay scaling を計算する
# https://arxiv.org/abs/2405.13698 の T_epoch framework を採用する
# 論文の中心的な考え方は T_epoch = B/(η·λ·D) を一定に保つこと。
# 上では学習率 scaling η ∝ √(B/B_ref) を使った。T_epoch を一定に保つためには、簡単な導出で次が必要になる:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# これらの論文が調べているのは AdamW であり、Muon ではない点に注意。
# ここでは AdamW の理論に従い、Muon でもおおむね機能することを期待している。
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(
        f"depth {args.depth} に合わせて weight decay を "
        f"{args.weight_decay:.6f} から {weight_decay_scaled:.6f} に変更します"
    )

# -----------------------------------------------------------------------------
# Optimizer を初期化 (MuonAdamW の組み合わせ: 行列 params は Muon、それ以外は AdamW)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# fp16 学習用の GradScaler (bf16/fp32 では不要。bf16 は fp32 と同じ exponent range を持つ)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("fp16 学習用に GradScaler を有効化しました")

# -----------------------------------------------------------------------------
# train/val DataLoader を初期化
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)


def build_val_loader():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="val",
        device=device,
    )


x, y, dataloader_state_dict = next(train_loader)  # 最初のデータ batch の load を開始

# -----------------------------------------------------------------------------
# 学習 iteration 数を計算し、各種 scheduler を準備する

# num_iterations: 明示指定、target flops、target data:param ratio の順にどれかから決める
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # 指定がある場合は num_iterations を明示値で上書き
    num_iterations = args.num_iterations
    print0(f"ユーザー指定の iteration 数を使用します: {num_iterations:,}")
elif args.target_flops > 0:
    # target flops から iteration 数を計算 (scaling law 解析で使用。例: runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"target FLOPs から iteration 数を計算しました: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # target param data ratio から iteration 数を計算 (最も一般的な用途)
    num_iterations = target_tokens // total_batch_size
    print0(f"target data:param ratio から iteration 数を計算しました: {num_iterations:,}")
else:
    raise ValueError("学習期間が指定されていません")
total_tokens = total_batch_size * num_iterations  # 実際に学習する token 数
print0(f"学習 token 総数: {total_tokens:,}")
print0(f"Tokens : Scaling params 比率: {total_batch_size * num_iterations / num_scaling_params:.2f}")  # 例: Chinchilla は約 20
print0(f"学習 FLOPs 推定総量: {num_flops_per_token * total_tokens:e}")

# 学習率 schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Muon optimizer の momentum scheduler (0.97 まで warmup し、LR warmdown 中に 0.90 へ warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Muon optimizer の weight decay scheduler (学習全体で 0 まで cosine decay)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# 学習 loop

# loop state (学習 loop 内で更新される変数)
if not resuming:
    step = 0
    val_bpb = None  # eval_every > 0 の場合に設定される
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # 学習 loss の EMA
    total_training_time = 0  # 学習の wall-clock 時間合計
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# 1 step あたりの目標 total batch size に到達するために必要な gradient accumulation micro-step 数を計算
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len  # single rank の 1 iteration あたり token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 全 rank 合計の 1 iteration あたり token 数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"rank ごとの micro-batch tokens: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"micro-batch tokens: {world_tokens_per_fwdbwd:,}")
print0(f"total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")


def make_checkpoint_metadata():
    return {
        "step": step,
        "val_bpb": val_bpb,  # 最後の step の loss
        "model_config": model_config_kwargs,
        "user_config": user_config,  # 学習 script への入力
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
        "total_batch_size": total_batch_size,
        "dataloader_state_dict": dataloader_state_dict,
        "loop_state": {  # 学習再開用の loop state (step 以外)
            "min_val_bpb": min_val_bpb,
            "smooth_train_loss": smooth_train_loss,
            "total_training_time": total_training_time,
        },
    }


# 開始
while True:
    last_step = step == num_iterations  # 最後に eval/save できるよう loop は num_iterations+1 回走る
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 定期的に val bpb を評価 (全 rank が参加)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"ステップ {step:05d} | 検証 bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # 定期的に CORE metric を推定 (全 rank が参加)
    # 入力 shape が変わり続けるため、未 compile の元モデルを使う
    # より一貫して正確な結果にするため、評価中は FP8 を無効化して BF16 を使う
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"ステップ {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # 定期的にモデルから sample を生成 (master process のみ)
    # 入力 shape が変わり続けるため、未 compile の元モデルを使う
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)  # 再 compile を避けるため orig_model を使う
        for prompt in SAMPLE_PROMPTS:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # checkpoint 保存: 実行の最後、または save_every ごと。ただし最初の step と resume step は除く
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # モデルパラメータ
            optimizer.state_dict(),  # optimizer の状態
            make_checkpoint_metadata(),
            rank=ddp_rank,
        )

    # 終了条件 (TODO: loss explosion なども追加できる)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # 単一の学習 step
    # gradient を評価
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()  # ログ用
        loss = loss / grad_accum_steps  # 各 .backward() は grad を加算するため、ここで loss を正規化
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader)  # GPU が forward/backward 中に次 batch を prefetch
    # optimizer を step
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # 分散学習では、step を skip するかどうかを全 rank で一致させる必要がある。
        # 各 rank は独立に inf/nan gradient に遭遇しうるため、found_inf flag を all-reduce する
        # (MAX = どれかの rank が inf を見つけたら全 rank が skip)。
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()  # .item() は CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # ログ (CPU のみの処理)
    ema_beta = 0.9  # 見やすいログにするための EMA decay factor
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f  # 学習 loss の EMA
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))  # EMA の bias 補正
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # 最初の 10 step 以降だけを計測対象にする
    # 1 step あたりの平均時間から ETA を計算 (最初の 10 step を除外)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(
        f"ステップ {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
        f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
        f"bf16_mfu: {mfu:.2f} | epoch: {epoch} | "
        f"合計時間: {total_training_time/60:.2f}m{eta_str}"
    )
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state 更新
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # garbage collector は少し過剰に動きがちで、よく分からない理由で頻繁に ~500ms ほど cycle scan に費やす。
    # しかも毎回ごく少数の小さな object しか掃除しないことが多い。
    # そのためここでは手動で管理して補助する。
    if first_step_of_run:
        gc.collect()  # setup 由来の garbage をまとめて手動回収
        gc.freeze()  # 現在生存している object をすぐ freeze し、GC 対象から外す
        gc.disable()  # 強い介入: 以降は基本的に GC を無効化
    elif step % 5000 == 0:  # 5000 step ごと
        gc.collect()  # 非常に長い run に備えて念のため手動回収

# 追加の統計を表示
print0(f"最大メモリ使用量: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"学習時間合計: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"最小 validation bpb: {min_val_bpb:.6f}")

# report にログ
from nanochat.report import get_report
get_report().log(section="ベースモデル学習", data=[
    user_config, # CLI args
    { # 学習設定の統計
        "パラメータ数": num_params,
        "1 トークンあたりの FLOPs": f"{num_flops_per_token:e}",
        "計算された iteration 数": num_iterations,
        "学習 token 数": total_tokens,
        "Tokens : Scaling params 比率": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # 学習結果の統計
        "最小 validation bpb": min_val_bpb if val_bpb is not None else None,
        "最終 validation bpb": val_bpb,
        "CORE metric 推定値": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "学習 FLOPs 合計": f"{flops_so_far:e}",
        "学習時間合計": f"{total_training_time/60:.2f}m",
        "最大メモリ使用量": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# 後片付け
wandb_run.finish() # wandb run を終了
compute_cleanup()
