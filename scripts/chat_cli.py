"""
前回からコードが大きく変わったため、新しく更新した chat mode です。

現時点では単一 GPU での実行を想定しています:
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='モデルとチャットします')
parser.add_argument('-i', '--source', type=str, default="sft", help="モデルの種別: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='読み込む model tag')
parser.add_argument('-s', '--step', type=int, default=None, help='読み込む step')
parser.add_argument('-p', '--prompt', type=str, default='', help='モデルへの prompt。単発の応答を返します')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='生成時の temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling パラメータ')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='評価に使う device type: cuda|cpu|mps。空なら自動検出')
args = parser.parse_args()

# モデルとトークナイザーを初期化

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# chat state machine 用の special token
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# 効率的な生成のために Engine を作成
engine = Engine(model, tokenizer)

print("\nNanoChat 対話モード")
print("-" * 50)
print("会話を終了するには 'quit' または 'exit' と入力してください")
print("新しい会話を開始するには 'clear' と入力してください")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # 起動コマンドから prompt を取得
        user_input = args.prompt
    else:
        # console から対話的に prompt を取得
        try:
            user_input = input("\nユーザー: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

    # special command を処理
    if user_input.lower() in ['quit', 'exit']:
        print("終了します。")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("会話をクリアしました。")
        continue

    if not user_input:
        continue

    # User message を会話に追加
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Assistant の生成を開始
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # batch 次元を外す (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    # assistant end token が最後の token になるよう保証する。
    # max tokens で生成が終わった場合でも、末尾に追加する必要がある。
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # prompt mode では単発の応答だけを返して終了する
    if args.prompt:
        break
