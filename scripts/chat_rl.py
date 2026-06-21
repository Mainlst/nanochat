"""
GSM8K に対して "GRPO" 風の強化学習を行います。

GRPO と引用符付きで呼んでいるのは、実際にはかなり単純化され、
REINFORCE に近いものになっているためです。

1) trust region を削除するため、reference model への KL regularization はありません
2) on-policy なので PPO ratio+clip は不要です
3) sequence-level ではなく token-level の DAPO 風 normalization を使います
4) z-score normalization (r - mu)/sigma の代わりに、advantage として (r - mu) だけを使います

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI 引数
parser = argparse.ArgumentParser(description="GSM8K に対する強化学習")
# ログ
parser.add_argument("--run", type=str, default="dummy", help="wandb run 名 ('dummy' で wandb ログを無効化)")
# 実行環境
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (空なら自動検出)")
# モデル読み込み
parser.add_argument("--model-tag", type=str, default=None, help="読み込む model tag")
parser.add_argument("--model-step", type=int, default=None, help="読み込む model step")
# 学習期間
parser.add_argument("--num-epochs", type=int, default=1, help="GSM8K を何 epoch 回すか")
# バッチサイズ / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="1 forward pass あたりの最大 batch size")
parser.add_argument("--examples-per-step", type=int, default=16, help="全 rank 合計の optimization step あたり example 数")
parser.add_argument("--num-samples", type=int, default=16, help="example/question あたりの sample 数")
# 生成
parser.add_argument("--max-new-tokens", type=int, default=256, help="sample あたりに生成する最大 token 数")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = 無効)")
# 最適化
parser.add_argument("--embedding-lr", type=float, default=0.2, help="embedding パラメータの学習率 (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="unembedding パラメータの学習率 (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="行列パラメータの学習率 (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="embedding/unembedding パラメータの weight decay (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="base LR に対する初期 LR の比率")
# 評価 / checkpoint
parser.add_argument("--eval-every", type=int, default=60, help="N step ごとに pass@k を評価")
parser.add_argument("--eval-examples", type=int, default=400, help="pass@k 評価に使う example 数")
parser.add_argument("--save-every", type=int, default=60, help="N step ごとに checkpoint を保存")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# compute/precision を初期化
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # この process が logging や checkpoint 保存などを行う

# wandb logging を初期化
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# モデルとトークナイザーを初期化
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # rollout sampling 用

# -----------------------------------------------------------------------------
# 学習用 example batch を yield する rollout / sampling generator loop

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"計算された step 数: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # padding 用にこの token を使ってよい。loss には使われない。
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # 各 rank は training data 内の異なる example を担当
    for example_idx in itertools.cycle(rank_indices):

        # まず user と assistant の両方を含む完全な conversation を取得
        conversation = train_task[example_idx]

        # conversation を token 化し、最後の Assistant message を削除して completion 用に Assistant を開始状態にする
        # (つまり <|assistant_start|> は残し、それ以降を削除する)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # batched generation で num_samples 個の sample を生成する。OOM 回避のため loop を使う
        model.eval() # model が eval mode であることを保証
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # OOM 回避のため順番に処理
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # sampling step ごとに seed を必ず変える
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # 各 sample の reward を計算
        rewards = []
        for sample_tokens in generated_token_sequences:
            # prompt 以降の生成 token だけを取得
            generated_tokens = sample_tokens[prefix_length:]
            # 生成応答を decode
            generated_text = tokenizer.decode(generated_tokens)
            # reward を計算
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # sequence 長 (time 次元) が揃うように padding
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # sequence と mask を PyTorch tensor に積む
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Transformer 用の autoregressive inputs と targets を生成
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # in-place modification を避けるため clone
        targets[mask_ids[:, 1:] == 0] = -1 # <-- ここで inplace modification。-1 は ignore index
        # Engine は prompt token と tool use token の両方に mask=0 を返す点にも注意。
        # そのため、prompt token や tool use の強制 token では (正しく) 学習しない。
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # z-score (x-mu)/sigma の代わりに、平均を引くだけで advantage を計算
        mu = rewards.mean()
        advantages = rewards - mu
        # inputs/targets は id の (B, T)、rewards は float の (B,) として yield
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# GSM8K pass@k 用の単純な評価 loop
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    GSM8K task を評価し、評価結果 record の list を返します。
    分散実行では全 rank が協調しますが、この関数は rank 間 reduction を行いません。
    それは caller の責務です。
    評価には時間がかかる場合があるため、この関数は record を 1 件ずつ yield します。
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Engine 内の batched generation で k 個の sample を生成
        assert num_samples <= args.device_batch_size # 通常は true。必要なら loop を追加できる。
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # 各 sample の正誤を確認
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # 以前もっと複雑な logging をしたかった名残で、少し大きめの record にしている。
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# 学習 loop

# optimizer を初期化
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# base learning rate に対する比率として初期 learning rate を設定
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: num_steps 全体で 0 まで単純に rampdown
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# desired examples_per_step を達成するために各 rank が担当する example 数を計算
print0(f"step あたりの総 sequence 数: {args.examples_per_step * args.num_samples}") # sequences/step 単位の total batch size
assert args.examples_per_step % ddp_world_size == 0, "希望する examples per step は rank 数で割り切れる必要があります"
examples_per_rank = args.examples_per_step // ddp_world_size # GPU ごと
print0(f"rank あたりの example 数: {examples_per_rank}")

# 学習 loop を開始
batch_iterator = get_batch()
for step in range(num_steps):

    # ときどきモデルを評価し、wandb にログする
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # k=1..device_batch_size の pass@k
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # 全 record を集める
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # record 総数で normalize
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # dataset 内の複数 example に対する rollout で forward/backward
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # training dataset の 1 example に対応する batch を取得
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # loss と gradient を評価
        model.train() # model が train mode であることを保証
        # device_batch_size を超えられないため、もう 1 つ loop が必要
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # この pass 用の batch を取り出す
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # log probability を計算。loss は NLL = -logp を計算するため、符号を反転する
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # PG objective を計算。ignore_index=-1 により invalid token の loss は 0 になる
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # valid token 数、pass 数、examples_per_rank で normalize
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # on-policy なので PPO ratio+clip を追加する必要はない
            # 最後に、最大化したい objective ではなく、最小化したい loss として定式化する
            loss = -pg_obj
            loss.backward()
            print0(f"step {step}/{num_steps} | example ステップ {example_step} | pass {pass_idx} | loss: {loss.item():.6f} | 平均 reward: {rewards.mean().item()}")
        # logging 用
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # この step の rollout 状況をまとめて logging
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # rank 間で集約
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"step {step}/{num_steps} | 平均 reward: {mean_reward} | 平均 sequence 長: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # モデルパラメータを更新
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # master process がときどきモデルを保存する。最初の step はスキップし、最後の step は保存する。
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # base model の depth を model tag の基準にする
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # GPTConfig の単純さに少し甘えている。TODO: もっときれいにする
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # optimizer state は保存しない
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ モデル checkpoint を {checkpoint_dir} に保存しました")

# report にログ
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI 引数
])

wandb_run.finish() # wandb run を終了
compute_cleanup()
