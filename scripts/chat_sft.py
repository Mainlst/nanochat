"""
モデルを教師あり fine-tuning (SFT) します。
実行方法:

python -m scripts.chat_sft

学習に torchrun を使う場合:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI 引数
parser = argparse.ArgumentParser(description="モデルを教師あり fine-tuning (SFT) します")
# ログ
parser.add_argument("--run", type=str, default="dummy", help="wandb run 名 ('dummy' で wandb ログを無効化)")
# 実行環境
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (空なら自動検出)")
# モデル読み込み
parser.add_argument("--model-tag", type=str, default=None, help="読み込む model tag")
parser.add_argument("--model-step", type=int, default=None, help="読み込む model step")
parser.add_argument("--load-optimizer", type=int, default=1, help="pretrained checkpoint から optimizer を warm-start (0=no, 1=yes)")
# 学習期間
parser.add_argument("--num-iterations", type=int, default=-1, help="optimization step 数 (-1 = full epoch)")
# バッチサイズ (デフォルト: pretrained checkpoint から継承)
parser.add_argument("--max-seq-len", type=int, default=None, help="最大 context 長 (デフォルト: pretrain から継承)")
parser.add_argument("--device-batch-size", type=int, default=None, help="デバイスごとの batch size (デフォルト: pretrain から継承)")
parser.add_argument("--total-batch-size", type=int, default=None, help="token 単位の total batch size (デフォルト: pretrain から継承)")
# 最適化 (デフォルト: pretrained checkpoint から継承)
parser.add_argument("--embedding-lr", type=float, default=None, help="embedding パラメータの学習率 (Adam) (デフォルト: pretrain から継承)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="unembedding パラメータの学習率 (Adam) (デフォルト: pretrain から継承)")
parser.add_argument("--matrix-lr", type=float, default=None, help="行列パラメータの学習率 (Muon) (デフォルト: pretrain から継承)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="base LR に対する初期 LR の比率")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="LR warmup に使う iteration の比率")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="LR warmdown に使う iteration の比率")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="初期 LR に対する最終 LR の比率")
# 評価
parser.add_argument("--eval-every", type=int, default=200, help="N step ごとに val bpb を評価 (-1 = 無効)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="val loss 評価に使う token 数")
parser.add_argument("--chatcore-every", type=int, default=200, help="N step ごとに ChatCORE metric を評価 (-1 = 無効)")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="ChatCORE のカテゴリ型タスクごとの最大問題数")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="ChatCORE の生成型タスクごとの最大問題数")
# データ混合
parser.add_argument("--mmlu-epochs", type=int, default=3, help="学習 mixture 内の MMLU epoch 数 (Multiple Choice を教える)")
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="学習 mixture 内の GSM8K epoch 数 (Math と Tool Use を教える)")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute 初期化
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"計算 dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | ピーク FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # CPU/MPS では MFU に意味がない

# wandb logging を初期化
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Flash Attention の状態
if not HAS_FA3:
    print0("警告: Flash Attention 3 が利用できないため、PyTorch SDPA fallback を使用します。学習効率は下がります。")

# モデルとトークナイザーを読み込む
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# 学習 hyperparameter を pretrained checkpoint から継承 (None = 継承、明示値 = 上書き)
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len",       2048,  meta),
    ("device_batch_size", 32,    meta),
    ("total_batch_size",  524288, meta),
    ("embedding_lr",      0.3,   pretrain_user_config),
    ("unembedding_lr",    0.004, pretrain_user_config),
    ("matrix_lr",         0.02,  pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"{name}={resolved} を pretrained checkpoint から継承しました")
    elif pretrain_val is not None and arg_val != pretrain_val:
        print0(f"注意: --{name.replace('_', '-')}={arg_val} は pretrained 値 {pretrain_val} を上書きします")
    else:
        print0(f"{name}={arg_val} を使用します")

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # 単一 rank の iteration あたり token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 全 rank 合計の iteration あたり token 数
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"rank ごとの micro-batch tokens: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"micro-batch tokens: {world_tokens_per_fwdbwd:,}")
print0(f"total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Optimizer を初期化 (combined MuonAdamW: 行列 params は Muon、それ以外は AdamW)
# pretraining では最後に weight_decay を 0 まで下げるため、SFT も 0 のまま続ける
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# 必要に応じて pretrained checkpoint から optimizer を warm-start する (momentum buffer など)
# 注意: load_state_dict は param_group metadata (LR, beta など) を pretrained 値で上書きする。
# pretraining warmdown により LR はほぼ 0 になっているため、読み込み後に新しい SFT LR を復元する。
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("pretrained checkpoint から optimizer state を読み込みました (momentum buffer のみ、LR は reset)")
    else:
        print0("警告: optimizer checkpoint が見つからないため、新しい optimizer で開始します (少し不利)")

# fp16 学習用の GradScaler (bf16/fp32 では不要)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("fp16 学習用に GradScaler を有効化しました")

# base learning rate に対する比率として初期 learning rate を上書き
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# SFT data mixture と DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_tasks = [
    SmolTalk(split="train"), # general conversation 460K rows
    CustomJSON(filepath=identity_conversations_filepath), # synthetic identity conversation 1000 rows
    CustomJSON(filepath=identity_conversations_filepath), # これを 2 epoch 分
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # epoch あたり 100K rows
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # epoch あたり 8K rows
    SimpleSpelling(size=200000, split="train"), # Simple Spelling 200K rows (例: spell the word 'apple')
    SpellingBee(size=80000, split="train"), # Spelling Bee 80K rows (例: how many 'r' are in 'strawberry'?)
]
train_dataset = TaskMixture(train_tasks)
print0(f"学習 mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
val_dataset = TaskMixture([
    SmolTalk(split="test"), # test set 24K rows
    MMLU(subset="all", split="test", stop=5200), # test set 14K rows。train ratio に合わせて 5.2K だけ使う
    GSM8K(subset="main", split="test", stop=420), # test set 1.32K rows。train ratio に合わせて 420 だけ使う
]) # 合計: 24K + 5.2K + 0.42K ~= 29.6K rows
# DataLoader はここで定義し、inputs, targets: shape (device_batch_size, max_seq_len) の 2D tensor を出す
# 最終的な num_iterations が事前には分からないのが大きな問題。そのため、以下 2 つの global variable を作り、
# data generator の中から更新する。
last_step = False # training dataset の末尾に到達したら True に切り替える
approx_progress = 0.0 # epoch の進行に合わせて 0 から 1 へ進む
current_epoch = 1 # logging 用に epoch を追跡
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    bestfit-pad packing を使う、SFT 用の BOS-aligned dataloader です。

    batch の各 row は BOS (conversation の開始) から始まります。
    conversation は best-fit algorithm で pack されます。収まる conversation がない場合、
    token を捨てないように crop ではなく padding します。
    padding 位置の target は -1 (cross-entropy の ignore_index) で mask されます。
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split は 'train' または 'val' である必要があります"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # 最後の位置の target 用に +1
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: (token_ids, loss_mask) tuple の list
    conv_buffer = []
    cursor = ddp_rank  # 各 rank が異なる conversation を処理 (fetching 用)
    consumed = ddp_rank  # buffering とは別に実際の消費量を追跡
    epoch = 1
    it = 0  # iteration counter

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # last_step は fetching ではなく consumption に基づいて発火する

    while True:
        rows = []
        mask_rows = []
        row_lengths = []  # 各 row の実 content 長 (padding を除く) を追跡
        for _ in range(args.device_batch_size):
            row = []
            mask_row = []
            padded = False
            while len(row) < row_capacity:
                # buffer に conversation があることを保証
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # 全体が収まる最大の conversation を探す
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # 収まる conversation が見つかったので丸ごと使う
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size  # 実際の消費量を追跡
                else:
                    # 収まる conversation がないため、crop せず残りを padding する
                    # これにより token を一切捨てない
                    content_len = len(row)
                    row.extend([bos_token] * remaining)  # BOS token で padding
                    mask_row.extend([0] * remaining)
                    padded = True
                    break  # Row is now full (with padding)

            # content 長を追跡: padding なしなら full row、padding ありなら padding 前の長さ
            if padded:
                row_lengths.append(content_len)
            else:
                row_lengths.append(row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # num_iterations が指定されている場合に尊重する停止条件
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # 進捗追跡を更新 (buffering を考慮し、cursor ではなく consumed に基づく)
        if split == "train":
            current_epoch = epoch
            if args.num_iterations > 0:
                approx_progress = it / args.num_iterations
            else:
                approx_progress = consumed / dataset_size
            # cursor が wrap したときではなく、十分に消費したときに last_step を発火
            if consumed >= dataset_size:
                last_step = True

        # tensor を構築
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # render_conversation 由来の loss mask を適用する (assistant completion は mask=1、
        # user prompt, BOS, special token, tool output は mask=0)。mask[1:] は
        # 1 つ shift された targets と揃う。mask されない位置は -1 (ignore_index) にする。
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1

        # targets 内の padding 位置を mask する (-1 = ignore_index)
        # 各 row について、targets の positions >= (content_length - 1) を mask する
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len-1:] = -1

        yield inputs, targets

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # epoch の進行に合わせて 0 から 1 へ進む

# 学習率 schedule (linear warmup, constant, linear warmdown)
# base_train と同じ形だが、SFT は num_iterations を事前に常に知っているとは限らないため、
# absolute step count の代わりに progress (0→1) を使う (dataset-driven stopping)。
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Muon optimizer 用の momentum scheduler
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# 学習 loop
x, y = next(train_loader) # 最初の data batch を先読み
min_val_bpb = float("inf")
smooth_train_loss = 0 # training loss の EMA
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # 学習の wall-clock time 合計
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # 分散実行での hang を避けるため、全 rank で last_step を同期
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # 定期的に val bpb を評価 (全 rank が参加)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | 検証 bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 定期的に ChatCORE metric を推定 (全 rank が参加)
    # 入力 shape が変わり続けるため、未 compile の元モデルを使う
    chatcore_results = {}
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        baseline_accuracies = {
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
        }
        task_results = {}
        for task_name in all_tasks:
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            max_problems = None if limit < 0 else limit  # -1 は制限なし
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                batch_size=args.device_batch_size, max_problems=max_problems)
            task_results[task_name] = acc
            print0(f"  {task_name}: {100*acc:.2f}%")
        # ChatCORE metric を計算 (centered accuracy の平均。0=random から 1=perfect の範囲)
        def centered_mean(tasks):
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        chatcore = centered_mean(all_tasks)
        chatcore_cat = centered_mean(categorical_tasks)
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "chatcore_metric": chatcore,
            "chatcore_cat": chatcore_cat,
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        })
        model.train()

    # 実行の最後に checkpoint を保存 (全 rank が参加し、それぞれ optimizer shard を保存する)
    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # 例: d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # 最後の step の loss
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config, # training script への入力
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # 単一の学習 step
    # gradient を評価
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # logging 用
        loss = loss / grad_accum_steps # 各 .backward() は grad sum なので、ここで loss を normalize
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(train_loader) # GPU が forward/backward で busy な間に次 batch を先読み
        progress = max(progress, approx_progress) # progress は単調増加だけにする
    # optimizer を step
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 状態更新
    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # training loss の EMA
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # EMA の bias を補正
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # 最初の 10 step 以降の時間だけを数える
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | 合計時間: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

    # garbage collector は cycle scan に頻繁に ~500ms ほど費やす。
    # 学習中の pause を避けるため、ここでは手動で管理する。
    if step == 1:
        gc.collect() # setup で出た大量の garbage を手動回収
        gc.freeze() # 現在生き残っている全 object を freeze し、GC から除外
        gc.disable() # 以下を除いて GC を完全に無効化
    elif step % 5000 == 0: # 5000 step ごと
        gc.collect() # 長時間実行に備えて念のため手動回収

# 追加の統計を表示
print0(f"最大メモリ使用量: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"学習時間合計: {total_training_time/60:.2f}m")
print0(f"最小 validation bpb: {min_val_bpb:.4f}")

# report にログ
from nanochat.report import get_report
get_report().log(section="SFT", data=[
    user_config, # CLI 引数
    { # 学習 setup に関する統計
        "iteration 数": step,
        "DDP world size": ddp_world_size,
    },
    { # 学習結果に関する統計
        "最小 validation bpb": min_val_bpb,
    }
])

# 後片付け
wandb_run.finish() # wandb run を終了
compute_cleanup()
