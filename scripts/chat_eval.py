"""
Chat モデルを評価します。
汎用的なコードはこのファイルに置き、評価ごとのコードは
nanochat ディレクトリから import します。

実行例:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
from functools import partial
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# 生成型評価 loop (1 問ずつ sample して評価)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # 評価を実行
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # prompt をトークン化
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # completion を取得
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # completion をテキストとして decode
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # 成功条件を評価
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # 統計を保持
        total += 1
        num_passed += int(passed)

        # ログ表示 (console の同じ行を上書き)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # 最終 summary の前に、上書き表示していた進捗行を改行で閉じる
    print()

    # 全 rank の結果を集約
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"最終結果: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # 正解率を返す
    return num_passed/total

# -----------------------------------------------------------------------------
# カテゴリ型評価 loop
# sampling が不要なのでかなり単純です。batch 単位で処理し、
# 正解選択肢の logits だけを確認します。

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # BOS を pad token として使ってよい。これらの位置は無視される

    # sampling が不要なので、独立した問題を batch 単位で処理する
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # 評価を実行
    letter_to_id_cache = {} # 同じ文字が何度も出るので、tokenizer の処理を少し節約する
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # 問題 batch を準備する。長さが異なることがあるので pad/collate する。
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: この仕組みを作り直す
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # 最後の token の位置 (予測回答の位置)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # conversation batch 全体の logits を並列に取得 (ここで効率が上がる)
        with torch.no_grad():
            logits = model(prompt_ids) # (B, T, V)

        # 選択肢に対応する文字だけに絞って、利用可能な回答を評価する。
        # 利用可能な文字だけに焦点を絞るので、評価がかなり簡単になる。
        # より難しい代替案は Assistant から通常生成し、正しい文字 (例: A, B, C, D) で
        # 応答したかを確認する方法だが、評価ではよくこのようにタスクを簡単にする。
        for idx, conversation in enumerate(conversations):
            # この問題で利用可能な全文字の token id を取得
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "各文字は単一 token である必要があります"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # logits を回答位置と利用可能な回答文字だけに絞る
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # argmax 文字 (予測回答) を取得
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # 結果を評価
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # 全 rank の結果を集約
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"最終結果: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None):
    # 評価 object を作成
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    # 評価を実行
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"未対応のタスク評価タイプです: {task_object.eval_type}")
    return acc

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # コマンドライン引数を解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="モデルの種別: sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="タスク名。デフォルトは全タスク。複数タスクは | で区切る。")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='カテゴリ型評価のバッチサイズ')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='読み込む model tag')
    parser.add_argument('-s', '--step', type=int, default=None, help='読み込む step')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='評価する最大問題数')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='評価に使う device type: cuda|cpu|mps。空なら自動検出')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # 評価対象タスクを取得
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # 4 択の多肢選択 => 25%
        'ARC-Challenge': 0.25, # 4 択の多肢選択 => 25%
        'MMLU': 0.25, # 4 択の多肢選択 => 25%
        'GSM8K': 0.0, # 自由回答 => 0%
        'HumanEval': 0.0, # 自由回答 => 0%
        'SpellingBee': 0.0, # 自由回答 => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # 全タスク評価を順番に実行
    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            model, tokenizer, engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{task_name} 正解率: {100 * acc:.2f}%")

    # report にログ
    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # 可能なら ChatCORE metric を計算する (CORE と同様、centered accuracy の平均)
    # これにより、ChatCORE は 0 (ランダムベースライン) から 1 (最高性能) の範囲になる
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    get_report().log(section="Chat 評価 " + args.source, data=[
        vars(args), # CLI 引数
        results,
        chatcore_metric_dict,
    ])

    compute_cleanup()
