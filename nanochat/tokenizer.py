"""
GPT-4スタイルのBPEトークナイザーです。

以下の2つの実装オプションがあります：
1) HuggingFaceのトークナイザー（学習と推論の両方が可能ですが、操作がやや複雑です）
2) 独自のRustBPEトークナイザー（学習用）とtiktoken（効率的な推論用）
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # すべての文書はSequence Begin (BOS) トークンで始まり、文書の区切りとして機能します
    "<|bos|>",
    # 以下のトークンはファインチューニング時のみ使用され、会話内容をトークンIDに変換するために用いられます
    "<|user_start|>"  # ユーザーメッセージ
    "<|user_end|>",
    "<|assistant_start|>"  # アシスタントメッセージ
    "<|assistant_end|>",
    "<|python_start|>"  # アシスタントがPython REPLツールを呼び出す際のトークン
    "<|python_end|>",
    "<|output_start|>"  # Python REPLの出力をアシスタントに返す際のトークン
    "<|output_end|>",
]

# 注意：この分割パターンはGPT-4とは異なり、\p{N}{1,3}ではなく\p{N}{1,2}を使用しています
# この変更を行った理由は、語彙サイズが小さい場合に数字トークンを「無駄に」消費しすぎないようにするためです
# 語彙サイズ32Kの場合、2が最適な値であることを確認しました。1はやや劣り、3はさらに性能が低下します
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# HuggingFace Tokenizerをベースにした汎用GPT-4スタイルのトークナイザー
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """HuggingFace Tokenizerの軽量ラッパークラス（各種ユーティリティ機能を追加）"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # HuggingFaceの事前学習済みトークナイザーから初期化（例："gpt2"）
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # ローカルディスク上のディレクトリから初期化（例："out/tokenizer"）
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # テキストイテレータからトークナイザーを訓練
        # HuggingFace Tokenizerの設定
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # 必須設定
            unk_token=None,
            fuse_unk=False,
        ))
        # 正規化処理：なし
        tokenizer.normalizer = None
        # 事前トークナイザー：GPT-4スタイル
        # GPT-4がBPE処理前にテキストを分割する際に使用する正規表現パターン
        # 注意：このパターンは\p{N}{1,3}から\p{N}{1,2}に変更した。これは、非常に小規模なモデルや語彙サイズが小さい場合、
        # トークン空間の無駄遣いとなるため有害である可能性があると判断したためである
        # （ただしこの検証は未実施！ 今後の課題）
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingfaceではRegexオブジェクトで囲む必要がある!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # デコーダ：ByteLevel（ByteLevel事前トークン化器とペアで使用する）
        tokenizer.decoder = decoders.ByteLevel()
        # ポストプロセッサ：なし
        tokenizer.post_processor = None
        # トレーナー：BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # 最小頻度制限なし
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # 学習処理を開始
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # 単一の文字列をエンコードする
        # prepend/append は特殊トークン文字列またはトークン ID のいずれかを指定可能
        # num_threads は無視される（並列エンコード用に nanochat Tokenizer でのみ使用）
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # 特殊トークンを完全一致方式でエンコードする
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # HuggingFace の各モデルで BOS トークンは異なり、一貫性がほとんどない
        # 1) <|bos|> トークンが存在するか確認する
        bos = self.encode_special("<|bos|>")
        # 2) 見つからない場合、<|endoftext|> トークンを探す（GPT-2 モデルなど）
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) 上記でも見つからない場合は、エラーを発生させる方が無難である
        assert bos is not None, "トークナイザー内で BOS トークンを検出できませんでした"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"無効な入力型です: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # トークナイザーをディスクに保存する処理
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"トークナイザーを {tokenizer_path} に保存しました")

# -----------------------------------------------------------------------------
# rustbpe + tiktoken を組み合わせたトークナイザー
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """効率的な推論処理向けの tiktoken ラッパー（ただし学習には rustbpe を使用）"""
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) rustbpe を使用して学習
        tokenizer = rustbpe.Tokenizer()
        # 特殊トークンは __init__ メソッド内で後から挿入するため、ここでは学習しない
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special は少なくとも 256 である必要があります、取得値: {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) 推論用の対応する tiktoken エンコーディングを構築
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (トークンバイト列 -> マージ優先度ランク)
            special_tokens=special_tokens,  # dict[str, int] (特殊トークン名 -> トークン ID)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktokenではこの特殊文書区切りトークンを "<|endoftext|>" と呼んでいる
        # これは確かに紛らわしい命名だが、このトークンはほぼ常に文書の先頭に付加される
        # 主に推論時にLLMに対して新しいシーケンスの開始を通知するために使用される
        # そのためnanoChatでは一貫して "<|bos|>"（sequenceのbeginningの略）を使用しているが、歴史的には "<|endoftext|>" と呼ばれることが多い
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)
    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # textは文字列または文字列のリストのいずれかである

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: ここでの処理は若干非効率か？ :( うーん
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: 同様の問題
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"無効な入力型です: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # エンコーディングオブジェクトをディスクに保存
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"トークン化辞書を {pickle_path} に保存しました")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        単一のチャット会話（本システムでは「ドキュメント」と呼びます）をトークン化します。
        戻り値:
        - ids: list[int] はレンダリングされた会話のトークンIDリスト
        - mask: list[int] は同一長のリストで、Assistantが学習対象とするトークンには1が設定されます
        """
        # 返却するidsとmask、およびそれらを構築するための補助関数
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 最初のメッセージがシステムメッセージの場合...
        # => 第2メッセージ（ユーザーメッセージ）と統合します
        if conversation["messages"][0]["role"] == "system":
            # 現時点では会話データの整形処理が必要です...
            conversation = copy.deepcopy(conversation) # 元データを変更しないようにコピーを作成
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "システムメッセージは必ずユーザーメッセージに続く必要があります"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"会話には1件以上のメッセージが必要です: {messages}"

        # 必要な特殊トークンをすべて取得します
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # これで会話をトークン化できます
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # 前提条件の妥当性チェックを行い、予期せぬ動作を防止
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"メッセージ {i} の役割は {message['role']} ですが、{must_be_from} である必要があります"

            # 内容は単純な文字列か、ツール呼び出しなどを含む部分リストのいずれかです
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "ユーザーメッセージは文字列であることが期待されます"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # 単純な文字列の場合、単にトークンを追加します
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # テキスト部分の場合、単にトークンを追加します
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # Pythonツール呼び出しの場合、<|python_start|>と<|python_end|>タグ内にトークンを追加します
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # Python出力の場合、<|output_start|>と<|output_end|>タグ内にトークンを追加します
                            # これらのトークンは教師データに含まれません（テスト時にPythonから生成されるため）
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"不明なパートタイプ: {part['type']}")
                else:
                    raise ValueError(f"不明なコンテンツタイプ: {type(content)}")
                add_tokens(assistant_end, 1)

        # 最大トークン数MAXで切り捨て（OOMエラー防止に有効）
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask
    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """デバッグ時に便利な補助関数: render_conversationのトークン化処理を可視化"""
        RED = '\033[91m'  # 赤
        GREEN = '\033[92m'  # 緑
        RESET = '\033[0m'  # リセット
        GRAY = '\033[90m'  # グレー
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        強化学習環境で使用される関数。この設定では、アシスタントに完了文を生成させるための会話をレンダリングする。
        チャットSFTケースとは異なり、マスク値を返す必要はない。
        """
        # 必要な修正処理: アシスタント側の最後のメッセージを削除する必要がある
        conversation = copy.deepcopy(conversation)  # 元のデータを変更しないようにコピーを作成
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "最後のメッセージは必ずアシスタントからのものでなければならない"
        messages.pop()  # アシスタント側の最後のメッセージをその場で削除

        # 次に会話をトークン化する
        ids, mask = self.render_conversation(conversation)

        # 最後に、アシスタントに完了文を生成させるためのトリガーとして、アシスタント開始トークンを追加する
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat専用の便利な関数

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"トークンバイトデータが {token_bytes_path} に存在しません。tok_train.py によって生成されるはずです"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
