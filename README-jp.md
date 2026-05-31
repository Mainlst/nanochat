# nanochat

![nanochatロゴ](dev/nanochat.png)
![スケーリング則](dev/scaling_laws_jan26.png)

nanochatは、大規模言語モデル（LLM）のトレーニング用に設計された最もシンプルな実験用フレームワークです。単一のGPUノード上で動作するように設計されており、コードは最小限に抑えられているためカスタマイズが容易です。トークン化、事前学習、ファインチューニング、評価、推論、チャットUIなど、LLM開発の主要な全工程を網羅しています。例えば、2019年に約43,000ドルのコストがかかったGPT-2レベルの能力を持つLLMを、わずか48ドル（8XH100 GPUノードで2時間程度）でトレーニング可能で、その後は使い慣れたChatGPT風のWeb UIで対話することができます。スポットインスタンスを使用すれば、総コストは約15ドル程度に抑えることも可能です。より一般的に説明すると、nanochatは単一の設定パラメータ`--depth`（GPTトランスフォーマーモデルの層数。GPT-2レベルの能力は約26層に相当）を設定するだけで、計算リソースを効率的に活用した一連のモデルを自動的に最適化してトレーニングできるように構成されています。その他のハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率調整、トレーニング期間、重み減衰など）はすべて最適な方法で自動的に計算されます。
リポジトリに関するご質問については、Devin/Cognitionが提供する[DeepWiki](https://deepwiki.com/karpathy/nanochat)を利用するか、[ディスカッションタブ](https://github.com/karpathy/nanochat/discussions)をご利用ください。また、Discordの[#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031)チャンネルに直接お越しいただくこともできます。

## GPT-2到達時間リーダーボード

現在の開発の主な焦点は、最も計算リソースを必要とする事前学習段階の最適化にあります。modded-nanogptリポジトリに着想を得て、開発チームはコミュニティの協力を促進するため、「GPT-2スピードラン」のリーダーボードを設置しています。これはDCLM COREスコアによって測定される、nanochatモデルがGPT-2レベルの能力を獲得するまでに必要な実時間を競うものです。[runs/speedrun.sh](runs/speedrun.sh)スクリプトは常に、GPT-2レベルのモデルをトレーニングして対話するための標準的な方法を反映しています。現在のリーダーボードの状況は以下の通りです：

| 順位 | 所要時間 | 検証用BPE精度 | COREスコア | 説明 | 日付 | コミット | 貢献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168時間 | - | 0.2565 | オリジナルのOpenAI GPT-2チェックポイント | 2019年 | - | OpenAI |
| 1 | 3.04時間 | 0.74833 | 0.2585 | d24ベースライン（若干過学習気味） | 2026年1月29日 | 348fbb3 | @karpathy |
| 2 | 2.91時間 | 0.74504 | 0.2578 | d26（若干未学習気味）**+fp8** | 2026年2月2日 | a67eba3 | @karpathy |
| 3 | 2.76時間 | 0.74645 | 0.2602 | 総バッチサイズを100万トークンに増加 | 2026年2月5日 | 2c062aa | @karpathy |
| 4 | 2.02時間 | 0.71854 | 0.2571 | データセットをNVIDIA ClimbMixに変更 | 2026年3月4日 | 324e69c | @ddudek @karpathy |
| 5 | 1.80時間 | 0.71808 | 0.2690 | autoresearch [第1ラウンド](https://x.com/karpathy/status/2031135152349524125) | 2026年3月9日 | 6ed7d1d | @karpathy |
| 6 | 1.65時間 | 0.71800 | 0.2626 | autoresearch 第2ラウンド | 2026年3月14日 | a825e63 | @karpathy |

私たちが特に重視する指標は「GPT-2達成時間」です。これは8XH100 GPUノードにおいて、GPT-2（16億パラメータ）のCOREスコアを上回るまでに必要な実時間を指します。GPT-2 COREスコアは0.256525です。2019年にGPT-2のトレーニングには約43,000ドルのコストがかかっていましたが、7年間でスタック全体にわたる多くの技術的進歩があったおかげで、現在でははるかに高速に、しかも100ドル以下のコストで実現可能になっています（例えば現在のGPU時間単価が約3ドルの場合、8XH100ノードは約24ドル/時間で、2時間で約48ドルとなります）。

リーダーボードの解釈方法や貢献方法についての詳細は、[dev/LEADERBOARD.md](dev/LEADERBOARD.md)をご覧ください。

## 導入手順

### 環境設定

nanochatでは依存関係管理に[uv](https://docs.astral.sh/uv/)を使用しています。インストール方法は以下の通りです：

```bash
uv sync --extra gpu    # CUDA対応GPU環境（A100/H100など）の場合
uv sync --extra cpu    # CPU専用環境またはMPSを使用する場合
source .venv/bin/activate
```

開発用環境のセットアップ（pytest、matplotlib、ipykernel、transformersなどの追加パッケージを含む）：

```bash
uv sync --extra gpu --group dev
```
### GPT-2のトレーニングと対話

最も楽しいのは、自分でGPT-2をトレーニングして対話することです。そのために必要な全パイプラインは、単一のファイル[runs/speedrun.sh](runs/speedrun.sh)にまとめられており、8XH100 GPUノード上での実行を想定して設計されています。お好みのプロバイダーから新しい8XH100 GPUインスタンスを起動し（私は[Lambda](https://lambda.ai/service/gpu-cloud)を愛用しています）、トレーニングスクリプトを実行してください：

```bash
bash runs/speedrun.sh
```

実行には約3時間かかるため、screenセッションで実行するのが便利です。トレーニングが完了したら、ChatGPT風のWeb UIを通じて対話できます。必ずローカルのuv仮想環境が有効になっていることを確認してください（`source .venv/bin/activate`を実行）。そして以下のようにサービスを起動します：
```bash
python -m scripts.chat_web
```

表示されたURLにアクセスしてください。例えばLambdaを使用する場合、ノードのパブリックIPアドレスにポート番号を加えたURL（例：[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)）にアクセスします。あとは通常どおりChatGPTに話しかけるようにLLMと対話してください！物語や詩を書かせたり、「あなたは誰ですか？」と質問して幻覚を起こさせてみたり、「空が青い理由」や「なぜ緑なのか」といった質問をしてみましょう。このスピードランモデルは4e19 FLOPsの計算能力を持つため、まるで幼稚園児と話しているような感覚になるかもしれません :)。
---

<img width="2672" height="1520" alt="画像" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

その他の注意点：

- このコードはAmpere 8XA100 GPUノードでも正常に動作しますが、若干速度が遅くなります。
- `torchrun`を省略すれば、単一GPU環境でもコードは完全に動作し、ほぼ同等の結果が得られます（コードは自動的に勾配蓄積モードに切り替わります）。ただし処理時間は8倍長くかかります。
- GPUのメモリ容量が80GB未満の場合、ハイパーパラメータを調整するか、OOMエラーやVRAM不足が発生する可能性があります。スクリプト内の`--device-batch-size`パラメータを探し、適切な値に調整してください（デフォルトの32から16、8、4、2、さらには1まで減らすことができます）。これより小さい値を使用する場合は、より高度な知識と工夫が必要になります。
- コードの大部分は一般的なPyTorch実装であるため、xpuやmpsなど、PyTorchをサポートするあらゆる環境で動作するはずです。ただし私はこれらのコードパスをすべて実際にテストしたわけではないので、場合によっては最適化が必要な箇所があるかもしれません。

## 研究関連

研究者の方でnanochatの改善にご協力いただける場合、特に注目すべきスクリプトは[runs/scaling_laws.sh](runs/scaling_laws.sh)と[runs/miniseries.sh](runs/miniseries.sh)です。関連するドキュメントについては[1月7日版ミニシリーズv1](https://github.com/karpathy/nanochat/discussions/420)を参照してください。簡単な実験（事前学習5分程度）を行う場合、私のおすすめは12層モデル（GPT-1サイズ相当）で学習させることです。例えば以下のように設定します：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

この設定ではwandbを使用し（実行名は「d12」）、最終ステップでのみCOREメトリクスを計測し、中間チェックポイントのサンプリングや保存は行いません。私はコードを変更した後、d12（またはd16など）を再実行して効果を確認するという反復作業をよく行います。実行結果を評価する際には、wandbのプロットで以下の項目を特に確認することをおすすめします：

1. `val_bpb`（検証損失をバイトあたりビット数という語彙サイズに依存しない単位で表したもの）を`step`、`total_training_time`、`total_training_flops`の関数としてプロットしたもの
2. `core_metric`（DCLM COREスコア）
3. VRAM使用率、`train/mfu`（モデルFLOPS使用率）、`train/tok_per_sec`（トレーニングスループット）

具体的な使用例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照してください。

重要なポイントとして、nanochatはトランスフォーマーの深度という単一のパラメータを中心に設計・実装されています。この単一の整数値によって、トランスフォーマーの幅、ヘッド数、学習率調整、トレーニング期間、重み減衰など、その他のすべてのハイパーパラメータが自動的に決定され、計算効率が最適化されたモデルが訓練されます。ユーザーはこれらの設定を意識せず、単に`--depth`オプションでモデルサイズを指定するだけで、すべてが「自動的に」適切に設定されます。深度を段階的に変更することで、様々なサイズで計算効率が最適化されたnanochatモデルのシリーズを構築できます。現在最も注目されているGPT-2能力モデルは、現行コードではd24からd26の範囲に該当します。ただし、リポジトリへの変更提案は、あらゆる深度設定で確実に動作するよう十分に原理的な根拠に基づいている必要があります。
## CPU/MPS環境での実行方法

[runs/runcpu.sh](runs/runcpu.sh)スクリプトは、CPUまたはApple Silicon環境での実行方法を示す非常にシンプルな例です。学習対象のLLMを大幅に縮小することで、数十分程度の合理的なトレーニング時間に収めています。この方法では強力な結果を得ることは期待できません。

## 精度/データ型

nanochatでは`torch.amp.autocast`は使用していません。代わりに、精度管理は単一のグローバル変数`COMPUTE_DTYPE`（`nanochat/common.py`で定義）を通じて明示的に制御されています。デフォルトでは、この値はハードウェア構成に基づいて自動的に検出されます：

| ハードウェア | デフォルトデータ型 | 理由 |
|--------------|------------------|------|
| CUDA SM 80以上 (A100, H100, ...) | `bfloat16` | ネイティブのbf16テンソルコアを使用 |
| CUDA SM 80未満 (V100, T4, ...) | `float32` | bf16非対応。`NANOCHAT_DTYPE=float16`を指定することでfp16を使用可能（GradScalerを利用） |
| CPU/MPS | `float32` | 低精度テンソルコアを搭載していないため |

デフォルト値は、環境変数`NANOCHAT_DTYPE`で上書き可能です：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # 強制的にfp32を使用
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # 強制的にbf16を使用
```

動作原理：モデルの重みは最適化アルゴリズムの精度確保のためfp32で保存されますが、独自の`Linear`レイヤーでは順伝播時に`COMPUTE_DTYPE`にキャストされます。埋め込み表現はメモリ節約のため直接`COMPUTE_DTYPE`形式で保存されます。これにより、自動キャストと同様の混合精度の利点を得つつ、各処理をどの精度で実行するかを完全に明示的に制御できます。

注記：`float16`での学習を行う場合、`base_train.py`内で自動的に`GradScaler`が有効化され、勾配のアンダーフローを防止します。SFTではこの機能もサポートされていますが、現在RL（強化学習）では未対応です。また、fp16での推論はどこでも正常に動作します。
## ガイド

これまでに公開したガイドの中から、特に参考になると思われるものを最新順からご紹介します：

- [2026年2月1日：100ドル未満でGPT-2を超える：nanochat開発の軌跡](https://github.com/karpathy/nanochat/discussions/481)
- [2026年1月7日 ミニシリーズ v1](https://github.com/karpathy/nanochat/discussions/420) では、最初のnanochatモデルシリーズの開発過程を詳細に記録しています。
- nanochatに新たな機能を追加する方法については、[ガイド：ストロベリーに含まれる「r」の数を数える方法（および一般的な機能追加方法）](https://github.com/karpathy/nanochat/discussions/164) をご覧ください。
- nanochatをカスタマイズする方法については、[ガイド：nanochatに個性を吹き込む方法](https://github.com/karpathy/nanochat/discussions/139) をご覧ください。このガイドでは、合成データ生成とそのSFT（Supervised Fine-Tuning）段階への統合を通じて、nanochatの性格特性を調整する方法について解説しています。
- [2025年10月13日：nanochatのオリジナル投稿](https://github.com/karpathy/nanochat/discussions/1) ではnanochatの紹介を行っていますが、現在では一部の情報が古くなっているほか、モデル自体も現在のマスターバージョンに比べてかなり古く、性能も劣っています。

## ファイル構成

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # アイデンティティ用のサンプル合成データ
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretrainingデータシャードの生成スクリプト
├── nanochat
│   ├── __init__.py                 # 空の初期化ファイル
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存/読み込み機能
│   ├── common.py                   # 各種補助ユーティリティ関数（利便性向上用）
│   ├── core_eval.py                # ベースモデルのCOREスコア評価機能（DCLM論文準拠）
│   ├── dataloader.py               # 分散データローダー用のトークン化処理
│   ├── dataset.py                  # Pretrainingデータのダウンロード/読み込みユーティリティ
│   ├── engine.py                   # KVキャッシュを活用した効率的なモデル推論処理
│   ├── execution.py                # LLMがPythonコードをツール実行可能にする機能
│   ├── gpt.py                      # GPT用のnn.Module実装Transformerモデル
│   ├── logo.svg
│   ├── loss_eval.py                # 損失値ではなくビット/バイト単位での評価機能
│   ├── optim.py                    # AdamW + Muon最適化アルゴリズム（1GPUおよび分散処理対応）
│   ├── report.py                   # nanochatレポート作成用ユーティリティ
│   ├── tokenizer.py                # GPT-4スタイルに準拠したBPEトークナイザラッパー
│   └── ui.html                     # nanochatフロントエンド用のHTML/CSS/JavaScript
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseriesトレーニング用スクリプト
│   ├── runcpu.sh                   # CPU/MPS環境での実行方法を示す簡易サンプル
│   ├── scaling_laws.sh             # スケーリング則に関する実験スクリプト
│   └── speedrun.sh                 # ~$100の予算でnanochat d20モデルをトレーニングするスクリプト
├── scripts
│   ├── base_eval.py                # ベースモデル評価：COREスコア、ビット/バイト単位評価、サンプル生成
│   ├── base_train.py               # ベースモデル：トレーニング処理
│   ├── chat_cli.py                 # チャットモデル：CLI経由での対話機能
│   ├── chat_eval.py                # チャットモデル：評価タスク実行
│   ├── chat_rl.py                  # チャットモデル：強化学習による学習
│   ├── chat_sft.py                 # チャットモデル：SFT（Supervised Fine-Tuning）によるトレーニング
│   ├── chat_web.py                 # チャットモデル：WebUI経由での対話機能
│   ├── tok_eval.py                 # トークナイザー：圧縮率評価
│   └── tok_train.py                # トークナイザー：トレーニング処理
├── tasks
│   ├── arc.py                      # 複数選択式の科学知識問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 任意のjsonl形式会話データからタスクを生成
│   ├── gsm8k.py                    # 8Kグレードの小学校算数問題
│   ├── humaneval.py                # 誤称；シンプルなPythonコーディング課題
│   ├── mmlu.py                     # 複数選択式問題、幅広い分野を網羅
│   ├── smoltalk.py                 # HF（Hugging Face）提供のSmolTalkデータセットを統合
│   └── spellingbee.py              # モデルに文字の綴りと数え方を学習させるタスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## 貢献について

nanochatの目的は、エンドツーエンドで1,000ドル未満の予算で運用可能なマイクロモデル分野における最先端技術の向上にあります。「アクセシビリティ」とは、単なる総コストだけでなく、認知的複雑さも含めた概念です。nanochatは網羅的に設定可能なLLM「フレームワーク」ではなく、巨大な設定オブジェクトやモデルファクトリー、複雑なif-then-else構造などはコードベースに含まれていません。単一で一貫性のある、最小限の構成でありながら可読性に優れ、容易に改変可能で、最大限のフォークが可能な「強力なベースライン」コードベースとして設計されており、最初から最後まで一貫して動作し、実際に対話可能なChatGPTモデルを生成することを目的としています。現時点で特に興味深いのは、GPT-2へのレイテンシを高速化すること（すなわちCOREスコアを0.256525以上に向上させること）です。現在は約3時間を要していますが、事前学習段階の改善によってさらに性能向上が見込めます。
現在のAIポリシー：情報開示。プルリクエストを提出する際には、LLMによる大幅な貢献部分や、自身が執筆していない部分、あるいは完全に理解していない部分について必ず明記してください。

## 謝辞

- プロジェクト名「nanochat」は、私が以前に手掛けた[nanoGPT](https://github.com/karpathy/nanoGPT)プロジェクトに由来しています。このプロジェクトでは事前学習のみを扱っていました。
- nanochatはまた、[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)からもインスピレーションを得ています。このプロジェクトではnanoGPTリポジトリをゲーム化し、明確な評価指標とリーダーボードを導入しており、その多くのアイデアと事前学習部分の実装を借用しています。
 
- ファインウェブとスモールトークの開発においては、[HuggingFace](https://huggingface.co/)のサポートに感謝いたします。
- 本プロジェクトの開発に使用した計算リソースについては、[Lambda](https://lambda.ai/service/gpu-cloud)に御礼申し上げます。
- 主要なLLM専門家であるAlec Radford氏（🧙‍♂️ アレク・ラドフォード）には、助言と指導に対して感謝いたします。
- nanochatリポジトリの管理責任者であるSofie [@svlandeg](https://github.com/svlandeg)氏には、issues管理、プルリクエスト処理、およびnanochatに関する議論の調整においてご協力いただいたことに感謝いたします。

## 引用方法

nanochatをご自身の研究で有用と感じられた場合、以下のように簡潔に引用してください：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: 予算100ドルで購入できる最高のChatGPT},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MITライセンス
