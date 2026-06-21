#!/usr/bin/env python3
"""
統合 web chat server です。単一の FastAPI instance から UI と API の両方を配信します。

data parallelism を使って、複数 GPU に request を分散します。各 GPU はモデルの完全なコピーを読み込み、
incoming request は利用可能な worker に分配されます。

起動例:

- 利用可能な単一 GPU (デフォルト)
python -m scripts.chat_web

- 4 GPUs
python -m scripts.chat_web --num-gpus 4

chat するには、console に表示された URL を開いてください。(cloud box 上なら public IP を使用してください)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - worker pool 状態付きの health check
  GET  /stats      - worker pool 統計と GPU 使用状況

濫用防止:
  - 1 request あたり最大 500 messages
  - 1 message あたり最大 8000 文字
  - 会話全体で最大 32000 文字
  - Temperature は 0.0-2.0 に制限
  - Top-k は 0-200 に制限 (0 は top-k filtering を無効化し、全語彙を使用)
  - Max tokens は 1-4096 に制限
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# 濫用防止の制限値
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 は top-k filtering を無効化し、全語彙を使う
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server を起動します')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='使用する GPU 数 (デフォルト: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="モデルの種別: sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='生成時のデフォルト temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='デフォルトの top-k sampling パラメータ')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='生成する最大 token 数のデフォルト')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='読み込む model tag')
parser.add_argument('-s', '--step', type=int, default=None, help='読み込む step')
parser.add_argument('-p', '--port', type=int, default=8000, help='server を起動する port')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='評価に使う device type: cuda|cpu|mps。空なら自動検出')
parser.add_argument('--host', type=str, default='0.0.0.0', help='server を bind する host')
args = parser.parse_args()

# 会話 traffic の logging を設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

@dataclass
class Worker:
    """特定の GPU にモデルを読み込んだ worker です。"""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    """それぞれ別 GPU にモデル replica を持つ worker pool です。"""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # 例: cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """各 GPU にモデルを読み込みます。"""
        print(f"{self.num_gpus} GPU で worker pool を初期化中...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "複数 worker/GPU は CUDA のみ対応です。cpu|mps では使用できません。"

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"GPU {gpu_id} にモデルを読み込み中...")
            else:
                device = torch.device(device_type) # 例: cpu|mps
                print(f"{device_type} にモデルを読み込み中...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"全 {self.num_gpus} worker の初期化が完了しました")

    async def acquire_worker(self) -> Worker:
        """pool から利用可能な worker を取得します。"""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """worker を pool に戻します。"""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """濫用防止のため chat request を検証します。"""
    # message 数を確認
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="少なくとも 1 件の message が必要です")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"message が多すぎます。1 request あたり最大 {MAX_MESSAGES_PER_REQUEST} 件です"
        )

    # 個別 message 長と会話全体の長さを確認
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} の content が空です")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} が長すぎます。1 message あたり最大 {MAX_MESSAGE_LENGTH} 文字です"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"会話全体が長すぎます。最大 {MAX_TOTAL_CONVERSATION_LENGTH} 文字です"
        )

    # role 値を検証
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} の role が無効です。'user' または 'assistant' である必要があります"
            )

    # temperature を検証
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature は {MIN_TEMPERATURE} から {MAX_TEMPERATURE} の範囲である必要があります"
            )

    # top_k を検証
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k は {MIN_TOP_K} から {MAX_TOP_K} の範囲である必要があります"
            )

    # max_tokens を検証
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens は {MIN_MAX_TOKENS} から {MAX_MAX_TOKENS} の範囲である必要があります"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時に全 GPU へモデルを読み込みます。"""
    print("nanochat モデルを GPU 群に読み込み中...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"server 準備完了: http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """chat UI を配信します。"""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # 同一 origin を使うよう API_URL を差し替える
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """favicon と header 用の NanoChat logo を配信します。"""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """assistant 応答を streaming で生成します。"""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # emoji などの multi-byte UTF-8 文字を正しく扱うため token を蓄積する
    accumulated_tokens = []
    # 最後に完全だった UTF-8 文字列 (replacement character なし) を追跡する
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]

        # 停止条件
        if token == assistant_end or token == bos:
            break

        # token を sequence に追加
        accumulated_tokens.append(token)
        # UTF-8 を正しく扱うため、蓄積済み token 全体を decode する
        # decode は基本的に table lookup と文字列連結なのでかなり効率的
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # replacement character で終わらない場合だけ text を emit する
        # これにより、不完全な UTF-8 sequence を emit しないようにする
        if not current_text.endswith('�'):
            # 前回の clean decode 以降に増えた text だけを取り出す
            new_text = current_text[len(last_clean_text):]
            if new_text:  # 新しい内容がある場合だけ yield
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """chat completion endpoint (streaming のみ)。multi-GPU 用に worker pool を使います。"""

    # 濫用防止の基本検証
    validate_chat_request(request)

    # incoming conversation を console にログ
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # pool から worker を取得 (すべて busy なら待機)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # conversation token を構築
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # streaming 応答。完了後に worker を release する
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # logging 用に応答を蓄積
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # assistant 応答を console にログ
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # streaming 完了後に worker を pool に戻す
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # error 時も必ず worker を release する
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """health check endpoint です。"""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """worker pool 統計を取得します。"""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"NanoChat Web Server を起動します")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, 最大 tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
