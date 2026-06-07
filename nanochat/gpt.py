"""
GPTモデル（再記述版、大幅に簡素化）
主な特徴：
- rotary embeddingsの採用（位置埋め込みは不使用）
- QK正規化の実装
- トークン埋め込み層とlm_headの重みを分離
- MLP内のrelu^2活性化関数
- トークン埋め込み層後の正規化処理
- rmsnormモジュールに学習可能なパラメータを設定しない
- 線形層におけるバイアス項の不使用
- 効率的な推論を実現するGroup-Query Attention（GQA）のサポート
- Flash Attention 3の統合実装
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Hopper+アーキテクチャでは自動的FA3を使用し、それ以外の場合はSDPAをフォールバックとして使用するカスタムFlash Attentionモジュール
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048  # 最大入力シーケンス長
    vocab_size: int = 32768   # 語彙サイズ
    n_layer: int = 12         # 全結合層の数
    n_head: int = 6           # クエリヘッドの数
    n_kv_head: int = 6        # キー/バリューヘッドの数（GQA用）
    n_embd: int = 768         # 埋め込み次元数
    # スライディングウィンドウ注意機構のパターン文字列。各層にタイル状に適用され、最終層は常にLを使用する
    # 文字の意味: L=long（完全なコンテキスト）、S=short（コンテキストの4分の1）
    # 使用例: "L"=全領域で完全なコンテキストを使用、"SL"=交互に使用、"SSL"=2つの短い領域の後に1つの長い領域を使用
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))  # 注意: この処理はbf16で実行されます。問題は発生しないようです

class Linear(nn.Linear):
    """入力データ型に合わせて重みをキャストするnn.Linearのカスタム実装
    autocastを置き換える機能。マスター重みは最適化器の精度維持のためfp32のまま保持されますが
    行列積演算は活性化データ型（通常は埋め込み層からのbf16）で実行されます"""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """GPTの特定層にValue Embeddingを適用する必要があるかどうかを返す関数
    （交互パターンで、最終層は必ず含まれる）"""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # マルチヘッド注意機構の場合
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # 最後の次元を2つの半分に分割
    y1 = x1 * cos + x2 * sin  # 次元ペアの回転処理
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # 入力を投影してクエリ、キー、値を抽出
        # 形状: (B, T, H, D) - FA3のネイティブなレイアウトで、転置は不要！
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 値残差処理（ResFormer方式）：各ヘッドごとに入力依存のゲートを用いて値埋め込みを混合
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # 形状: (B, T, n_kv_head)、値範囲: (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # クエリとキーにRotary Embeddingsを適用し、相対位置エンコーディングを生成
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QKのノルム計算
        q = q * 1.2  # より鋭い注意機構を実現（QとKのスケールを分離）、TODO: 最適な調整方法を検討
        k = k * 1.2
        # Flash Attentionの適用（Hopper+ではFA3、その他環境ではPyTorch SDPAをフォールバックとして使用）
        # window_sizeは(左境界, 右境界)のタプル: 因果的注意の場合は(N, 0)、全コンテキストの場合は(-1, 0)
        if kv_cache is None:
            # 訓練時: 因果的注意を適用し、オプションでスライディングウィンドウを使用可能
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # 推論時: キャッシュ管理を処理するflash_attn_with_kvcacheを使用
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # 最終層処理後に位置を進め、次の層の処理に備える
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # ヘッドを再構成し、残差ストリームに投影して戻す
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        重要な注意点: この __init__ 関数はメタデバイスコンテキストで実行されます (!!)
        したがって、この内部で行われる計算は形状とデータ型のみであり、実際のデータは含まれません。
        => 実際のデータ（パラメータ、バッファなど）の初期化は、init_weights() メソッドで別途行います。
        """
        super().__init__()
        self.config = config
        # スライディングウィンドウ注意機構用の層ごとのウィンドウサイズを計算
        # window_size は (左端, 右端) のタプル: 全コンテキストの場合は (-1, 0)、スライディングウィンドウの場合は (N, 0)
        self.window_sizes = self._compute_window_sizes(config)
        # 効率向上のためのパディング語彙の設定（DDP、テンソルコア対応）。こちらは最適化処理であり、実際の出力は forward() メソッド内で切り詰められます
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"効率向上のため、語彙サイズを {config.vocab_size} から {padded_vocab_size} にパディングします")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # レイヤーごとに学習可能なスカラーパラメータ（modded-nanogpt を参考に実装）
        # resid_lambdas: 各レイヤーで残差ストリームのスケールを調整（初期値 1.0 = 中立状態）
        # x0_lambdas: 各レイヤーで初期埋め込みをブレンドして再導入（初期値 0.0 = 無効化）
        # 別々のパラメータとして定義することで、異なるオプティマイザ設定が可能
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # 仮初期化、実際の初期化は init_weights() で実施
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # 仮初期化、実際の初期化は init_weights() で実施
        # Smear: 前トークンの埋め込み表現を現在トークンに混合（簡易的なビッグラム情報）
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: 最終正規化前に中間レイヤーのキャッシュされた残差を減算し、低レベル特徴を除去
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value 埋め込み表現（ResFormer スタイル）：交互レイヤーに適用し、最終レイヤーは常に含む
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # メタデバイス初期化に対応するため、ここで回転埋め込みを初期化するが、実際には「擬似」メタテンソルに過ぎない
        # 回転シーケンス長に関して、これらの回転埋め込みはメモリ使用量が非常に小さい/低コストであるため、
        # 10倍過剰に計算するが、実際にその量に達した場合はアサーションエラーを発生させる
        # 将来的にはキャッシュを動的に拡張可能だが、現時点ではこれで十分である
        self.rotary_seq_len = config.sequence_len * 10 # 過剰計算量10倍で十分だろう、TODO: より適切な実装を検討？
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # 持続的=Falseはチェックポイントに保存されないことを意味する
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        モデル全体の重み初期化をこの1つの関数にまとめることで、可読性を最大限に高めている

        wte（埋め込み層）：     正規分布（標準偏差=1.0）
        lm_head：               正規分布（標準偏差=0.001）
        各ブロックについて：
            attn.c_q：          一様分布（標準偏差=n_embdの平方根の逆数）
            attn.c_k：          一様分布（標準偏差=n_embdの平方根の逆数）
            attn.c_v：          一様分布（標準偏差=n_embdの平方根の逆数）
            attn.c_proj：       ゼロ値
            mlp.c_fc：          一様分布（標準偏差=n_embdの平方根の逆数）
            mlp.c_proj：        ゼロ値
        """

        # 埋め込み層と非埋め込み層の初期化
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # トランスフォーマーブロックの初期化：標準偏差が同じ正規分布と同じ範囲を持つ一様分布を使用
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # 標準偏差を正規分布と同じにするためのスケーリング因子
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # 外れ値を避けるため重みには一様分布を採用
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # 投影層の重みはゼロ初期化
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # c_fc層の重み初期化スケールを0.4倍に調整
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 層ごとのスカラーパラメータ
        # 層ごとの残差初期化：初期層では強い残差効果を、深層層では弱い残差効果を与える
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # 減衰するx0初期化：初期層ほど入力埋め込みのブレンド効果を強くする
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # スミア/バックアウトスカラーとスミアゲートは明示的に初期化する必要がある
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        # 値埋め込みの初期化（c_vと同様に、同じ標準偏差の一様分布を使用）
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # ゲート重みは小さい正の値で初期化し、ゲートが中立値より少し上からスタートするようにする
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # 回転埋め込み
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # 埋め込みを計算機用データ型にキャスト：最適化器は精度を落とした埋め込みを許容でき、メモリ節約になる
        # 例外：fp16を使用する場合、勾配スケーリングのために埋め込みはfp32でなければならない
        # なぜならGradScalerはfp16勾配を元のスケールに戻せないため
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: thetaの基数をさらに大きくすべきか？ 最近の傾向では100Kがより一般的である
        # モデルの埋め込みから自動的にデバイスを検出する
        if device is None:
            device = self.transformer.wte.weight.device
        # チャンネルをストライド処理する
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 時間ステップをストライド処理する
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 各(時間, チャンネル)ペアにおける回転周波数を計算する
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # 後のブロードキャスト用にバッチ次元とヘッド次元を追加
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        スライディングウィンドウ注意機構のための層ごとのウィンドウサイズを計算する。

        FA3のwindow_sizeパラメータに対する(左, 右)のタプルリストを返す:
        - left: 現在位置より前に注目するトークン数 (-1 = 無制限)
        - right: 現在位置より後に注目するトークン数 (因果的処理の場合は0)

        パターン文字列は層ごとに繰り返し適用される。最終層は常にL (完全なコンテキスト)を取得する。
        文字: L=long (完全なコンテキスト), S=short (コンテキストの4分の1)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"無効なwindow_pattern: {pattern}. 使用できるのはSとLのみです。"
        # 文字をウィンドウサイズにマッピング
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # FA3のタイルサイズに切り上げ (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # パターンを層ごとにタイル状に配置
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # 最終層は常に完全なコンテキストを取得
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        モデルの1トークンあたりの推定FLOPs数を返す（順方向処理＋逆方向処理）。
        各行列積重みパラメータは順方向処理で2FLOPs（乗算*と加算+）に寄与し、逆方向処理ではその2倍となるため、合計2+4=6FLOPsとなる。
        この計算について最も明確な説明はこちら：https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        さらに、12 * h * q * effective_seq_lenは、注意機構内のキー@クエリ行列積のFLOPs数を表している。
        スライディングウィンドウ方式では、effective_seq_lenは層ごとに異なり（ウィンドウサイズで上限が設定される）、
        参考：https://arxiv.org/abs/2204.02311（PaLM論文）
        これはChinchilla論文の厳密な計算式から約1%の誤差がある。この差異の理由は：
        - Chinchilla論文では埋め込み層の演算もFLOPsに含めて計算している（？ これは少し奇妙だが、単なるルックアップ処理なので無視している）
        - Chinchilla論文では注意機構のソフトマックス処理におけるexp/sum/divide演算もFLOPsに含めて計算している（これは少し疑わしいほど微小な値であるため、こちらも無視している）
        """
        nparams = sum(p.numel() for p in self.parameters())
        # 行列積演算に関与しないパラメータ（埋め込み層と層ごとのスカラー値）を除外
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # スライディングウィンドウを考慮した各層ごとの注意機構FLOPsの合計を計算
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) タプルのうち、left側を使用する
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        トークンあたりの総FLOPs数 = パラメータ数 - nparams_exclude × 6 + attn_flops
        return トークンあたりの総FLOPs数

    def num_scaling_params(self):
        """
        スケーリング則分析に必要な詳細なパラメータ数を返す。
        異なる研究論文ではパラメータの扱い方が異なる：
        - Kaplanらは埋め込みパラメータを除外して計算した
        - Chinchilla研究では全てのパラメータを含めて計算した
        参考文献：https://arxiv.org/abs/2203.15556（Chinchilla論文）
        参考文献：https://arxiv.org/abs/2001.08361（Kaplanらのオリジナルスケーリング則論文）

        各パラメータグループごとのカウントを辞書形式で返すため、下流の分析では
        どの組み合わせが最も明確なスケーリング則を示すかを実験的に検証できる。
        """
        # 各グループを個別にカウントする（setup_optimizersでのグループ分けと対応）
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "パラメータ数の不一致"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # すべてのパラメータをグループごとに分割
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # AdamWパラメータの学習率を ∝1/√dmodel でスケーリング（768次元モデル向けに調整済み）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"AdamWパラメータの学習率を ∝1/√({model_dim}/768) ＝ {dmodel_lr_scale:.6f} にスケーリングします")

        # 必要なフィールドをすべて明示的に設定したパラメータグループを構築
        param_groups = [
            # AdamWグループ（埋め込み層、言語モデルヘッド、スカラー値）
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # x0にはbeta1を大きめに設定
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muonグループ（行列パラメータを形状ごとにグループ化して積層）
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # 現在のシーケンス長に対応する回転埋め込みを取得（形状は (1, seq_len, 1, head_dim/2) である）
        assert T <= self.cos.size(1), f"シーケンス長が回転埋め込みキャッシュを超えています: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"回転埋め込みとidxが異なるデバイスに存在しています: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"回転埋め込みは{COMPUTE_DTYPE}型である必要があります、実際には{self.cos.dtype}型です"
        # kvキャッシュが存在する場合、回転埋め込みをキャッシュの現在位置にオフセットする必要がある
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # キャッシュを現在のシーケンス長に合わせて切り詰める

        # トークンを埋め込み処理
        x = self.transformer.wte(idx) # 現在のトークンを埋め込み
        x = x.to(COMPUTE_DTYPE) # 活性化が計算用データ型であることを確認（通常は不要な処理だが、fp16コードパスでは必須）
        x = norm(x)

        # スミア処理：前のトークンの埋め込みを現在位置に混合する（低コストなバイグラム情報）
        if kv_cache is None:
            # 訓練時／単純な生成時：全シーケンスが利用可能なので、高速なスライス処理を使用
            assert T > 1, "訓練時の順方向パスではT > 1である必要があります"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # kvキャッシュ推論時：キャッシュから前の埋め込みを読み込み、次のステップ用に現在の埋め込みを保存
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # プリフィル処理：1番目以降の位置にスミアを適用（訓練時と同様）
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # デコード時：単一トークンの場合、キャッシュされた前の埋め込みを使用
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Transformerの本体部分を処理
        x0 = x  # x0残差用に初期正規化埋め込みを保存
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # 中間点でキャッシュを保存
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # 中間層残差を減算して低レベル特徴を除去後、ロジット投影を実行
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # lm_headを処理してロジットを計算
        softcap = 15 # ロジットを[-softcap, softcap]の範囲に滑らかにクリップ
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- 非常に大きなテンソル、大量のメモリを消費
        logits = logits[..., :self.config.vocab_size] # パディング部分を削除するためスライス処理
        logits = logits.float() # ロジットのクリップ処理と損失計算のためにfp32に変換
        logits = softcap * torch.tanh(logits / softcap) # ロジットを圧縮

        if targets is not None:
            # 訓練時：ターゲットが与えられた場合、損失を計算して返す
            # TODO チャンク分割された交差エントロピー損失の実験を検討
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推論時：ロジットをそのまま返す
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        単純な自己回帰型ストリーミング推論手法
        簡略化のため、以下の前提条件を仮定する：
        - バッチサイズは1
        - idsと生成されるトークンは単純なPythonのリストおよび整数型である
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # バッチ次元を追加
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)に変形
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
