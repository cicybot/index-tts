import asyncio
import time
import uuid
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("/tmp/tasks")
TASK_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="IndexTTS TTS API")

# --------------------
# 内存任务队列
# --------------------
task_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

# --------------------
# 请求体模型
# --------------------
class TTSRequest(BaseModel):
    params: Dict[str, Any] = Field(
        ...,
        description="所有可控参数，直接传给 IndexTTS2.infer"
    )

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
        "status": "pending",
        "result": None,
        "worker_id": None,
        "submit_time": time.time(),
        "params": req.params
    }
    (TASK_FOLDER / f"{task_id}.json").write_text(json.dumps(task_data))
    await task_queue.put(task_id)
    return {"task_id": task_id}


@app.get("/tts/{task_id}")
def get_task(task_id: str):
    task_file = TASK_FOLDER / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    return json.loads(task_file.read_text())


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
    import worker  # 导入 worker.py 会自动启动 worker

    uvicorn.run(app, host="0.0.0.0", port=8000)
