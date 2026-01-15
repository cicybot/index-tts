import asyncio
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from pathlib import Path

# 内存队列，后续可改成 Redis
task_queue = asyncio.Queue(maxsize=32)
TASK_FOLDER = Path("./tmp")
TASK_FOLDER.mkdir(exist_ok=True)

app = FastAPI(title="IndexTTS Task Server")

# 请求体
class TTSRequest(BaseModel):
    text: str

# 发布任务接口
@app.post("/tts")
async def submit_tts(req: TTSRequest):
    if task_queue.full():
        raise HTTPException(status_code=429, detail="Server busy. Try later.")

    # 生成任务 ID
    task_id = str(uuid.uuid4())

    # 保存任务信息
    task_file = TASK_FOLDER / f"{task_id}.json"
    task_data = {"id": task_id, "text": req.text, "status": "pending", "result": None}
    task_file.write_text(json.dumps(task_data))

    # 发布到队列
    await task_queue.put(task_id)

    return {"task_id": task_id}

# 查询任务状态
@app.get("/tts/{task_id}")
def get_task(task_id: str):
    task_file = TASK_FOLDER / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(404, "Task not found")
    task_data = json.loads(task_file.read_text())
    return task_data


# ------------------------------
# 获取所有任务
# ------------------------------
@app.get("/tasks")
def get_all_tasks():
    tasks = []
    for task_file in TASK_FOLDER.glob("*.json"):
        try:
            task_data = json.loads(task_file.read_text())
            tasks.append(task_data)
        except Exception as e:
            print(f"[Warning] Failed to read {task_file}: {e}")
    return {"total": len(tasks), "tasks": tasks}


# 暴露队列（给 worker 导入使用）
@app.on_event("startup")
async def startup():
    app.state.task_queue = task_queue
    print("[startup] Task queue initialized.")

