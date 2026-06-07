"""
独自の BPE トークナイザーライブラリを使用してトークナイザーを学習します。
GPT-4 トークナイザーのスタイルに準拠しています。
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# コマンドライン引数の解析

parser = argparse.ArgumentParser(description='BPE トークナイザーの学習')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='学習に使用する最大文字数 (デフォルト: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='ドキュメントあたりの最大文字数 (デフォルト: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='語彙サイズ (デフォルト: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
# -----------------------------------------------------------------------------
# テキストイテレータ

def text_iterator():
    """
    1) バッチデータを単一のイテレータに平坦化する
    2) 各ドキュメントを args.doc_cap 文字の長さにトリミングする
    3) args.max_chars 文字に達した時点で処理を終了する
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()
# -----------------------------------------------------------------------------
# トークナイザーの学習
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"学習時間: {train_time:.2f}秒")

# -----------------------------------------------------------------------------
# # -----------------------------------------------------------------------------
# 追加処理: トークン ID からそのトークンのバイト数へのマッピングをキャッシュする
# これにより、バイトあたりビット数の効率的な評価が可能になる。一般的な平均損失とは異なり、
# この手法ではトークナイザーの語彙サイズに依存しない損失値を報告できる。
# 検証セットにおけるバイトあたりビット数は、我々が重視する主要な評価指標の一つとなる。
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # このトークンの Python 文字列表現
    if token_str in special_set:
        token_bytes.append(0) # 特殊文字はカウントしない
    else:
        id_bytes = len(token_str.encode("utf-8")) # このトークンを構成するバイト数
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"token_bytes を {token_bytes_path} に保存しました")
# レポート用ログ情報
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="トークナイザー学習", data=[
    vars(args), # argparse コマンドライン引数
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
