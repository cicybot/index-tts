import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uuid
import json
from indextts.infer_v2 import IndexTTS2
import torch

# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("./tasks")
TASK_FOLDER.mkdir(exist_ok=True)

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="IndexTTS TTS API")

# --------------------
# 内存任务队列
# --------------------
task_queue: asyncio.Queue = asyncio.Queue(maxsize=16)

# --------------------
# Worker ID
# --------------------
WORKER_ID = str(uuid.uuid4())

# --------------------
# 初始化模型（全局只加载一次）
# --------------------
tts_model = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False
)
print("[API] Model loaded, warming up...")

# warmup
tts_model.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="Warmup",
    output_path=str(TASK_FOLDER / "warmup.wav"),
    verbose=False
)
torch.cuda.synchronize()
print("[API] Warmup done.")


# --------------------
# 请求体模型
# --------------------
class TTSRequest(BaseModel):
    text: str
    spk_audio_prompt: str = None
    emo_vector: list = None
    use_random: bool = False


# --------------------
# 后台 worker
# --------------------
async def tts_worker():
    while True:
        task_id = await task_queue.get()
        task_file = TASK_FOLDER / f"{task_id}.json"
        if not task_file.exists():
            task_queue.task_done()
            continue

        task_data = json.loads(task_file.read_text())
        task_data["status"] = "running"
        task_data["worker_id"] = WORKER_ID
        task_data["start_time"] = time.time()
        task_file.write_text(json.dumps(task_data))

        try:
            # 执行 TTS
            wav = await asyncio.to_thread(
                tts_model.infer,
                spk_audio_prompt=task_data.get("spk_audio_prompt", None),
                text=task_data.get("text"),
                output_path=str(TASK_FOLDER / f"{task_id}.wav"),
                emo_vector=task_data.get("emo_vector", None),
                use_random=task_data.get("use_random", False),
                verbose=True
            )

            task_data["status"] = "done"
            task_data["result"] = str(TASK_FOLDER / f"{task_id}.wav")
            task_data["end_time"] = time.time()
            task_data["duration"] = task_data["end_time"] - task_data["start_time"]
            task_file.write_text(json.dumps(task_data))
            print(f"[Worker] Task {task_id} done in {task_data['duration']:.2f}s")

        except Exception as e:
            task_data["status"] = "error"
            task_data["error"] = str(e)
            task_data["end_time"] = time.time()
            task_data["duration"] = task_data["end_time"] - task_data.get("start_time", task_data["end_time"])
            task_file.write_text(json.dumps(task_data))
        finally:
            task_queue.task_done()
            await asyncio.sleep(0.01)


# --------------------
# API 接口
# --------------------
@app.post("/tts")
async def submit_tts(req: TTSRequest):
    if task_queue.full():
        raise HTTPException(status_code=429, detail="Server busy")

    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "text": req.text,
        "spk_audio_prompt": req.spk_audio_prompt,
        "emo_vector": req.emo_vector,
        "use_random": req.use_random,
        "status": "pending",
        "result": None,
        "submit_time": time.time()
    }
    (TASK_FOLDER / f"{task_id}.json").write_text(json.dumps(task_data))
    await task_queue.put(task_id)
    return {"task_id": task_id}


@app.get("/tts/{task_id}")
def get_task(task_id: str):
    task_file = TASK_FOLDER / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    task_data = json.loads(task_file.read_text())
    return task_data


@app.get("/tasks")
def get_all_tasks():
    tasks = []
    for task_file in TASK_FOLDER.glob("*.json"):
        try:
            tasks.append(json.loads(task_file.read_text()))
        except:
            continue
    return {"total": len(tasks), "tasks": tasks}


# --------------------
# __main__ 入口
# --------------------
if __name__ == "__main__":
    import uvicorn
    import time

    # 启动 worker
    loop = asyncio.get_event_loop()
    loop.create_task(tts_worker())

    # 启动 uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
