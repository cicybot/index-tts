import asyncio
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from indextts.infer import IndexTTS
import numpy as np

# ==============================
# FastAPI 应用
# ==============================
app = FastAPI(title="IndexTTS High-Concurrency Demo", version="0.1.0")

# ==============================
# 全局 TTS 模型和队列
# ==============================
tts_model: IndexTTS = None
tts_queue: asyncio.Queue = asyncio.Queue(maxsize=16)  # 并发限制

# ==============================
# 请求体模型
# ==============================
class TTSRequest(BaseModel):
    text: str

# ==============================
# 后台 Worker
# ==============================
async def tts_worker():
    global tts_model
    while True:
        task = await tts_queue.get()
        try:
            text = task["text"]
            future = task["future"]

            # 放到线程池执行，避免阻塞事件循环
            wav = await asyncio.to_thread(tts_model.tts, text)
            future.set_result(wav)

        except Exception as e:
            future.set_exception(e)
        finally:
            tts_queue.task_done()

# ==============================
# 启动事件
# ==============================
@app.on_event("startup")
async def startup():
    global tts_model
    print("[startup] Loading IndexTTS model...")
    tts_model = IndexTTS()
    print("[startup] Model loaded.")

    # 启动 1 个后台 worker
    asyncio.create_task(tts_worker())
    print("[startup] Worker started.")

# ==============================
# POST /tts 接口
# ==============================
@app.post("/tts")
async def tts_api(req: TTSRequest):
    if tts_queue.full():
        # 并发排队已满 → 返回 429
        raise HTTPException(status_code=429, detail="TTS server busy. Try again later.")

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # 将任务放入队列
    await tts_queue.put({
        "text": req.text,
        "future": future
    })

    # 等待 worker 执行完
    wav = await future

    # 返回 base64，方便前端直接播放
    wav_bytes = wav.astype(np.float32).tobytes()
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    return {"samples": len(wav), "wav_base64": wav_b64}
