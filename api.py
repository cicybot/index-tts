import asyncio
import time
import uuid
import json
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


TASK_FOLDER = Path("./frontend/dist")
TASK_FOLDER.mkdir(parents=True, exist_ok=True)
# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("./tasks")
TASK_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------
# 静态文件目录（供 /media 使用）
# --------------------
MEDIA_FOLDER = Path("./tasks")
MEDIA_FOLDER.mkdir(parents=True, exist_ok=True)
# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="IndexTTS TTS API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
# app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


class TTSRequest(BaseModel):
    params: Dict[str, Any]


class TasksResponse(BaseModel):
    total: int
    tasks: List[Dict[str, Any]]


# --------------------
# Frontend served at /
# API docs at /docs
# --------------------


# --------------------
# API 接口
# --------------------


# --------------------
# TTS提交接口
# --------------------
@app.post(
    "/tts",
    summary="Submit TTS task",
    description="""
提交TTS任务。`params` 参数会直接传给 `IndexTTS2.infer()`。  

**可传字段示例**：

- `text`: str, 要合成的文本  
- `spk_audio_prompt`: str, 参考音频路径，用于克隆说话人  
- `emo_vector`: list[float], 8维情感向量 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`  
- `cfg_value`: float, LM引导强度  
- `inference_timesteps`: int, 推理步数  
- `normalize`: bool, 是否启用外部文本归一化  
- `denoise`: bool, 是否启用降噪  
- `retry_badcase`: bool, 是否重试坏案例  
- `retry_badcase_max_times`: int, 最大重试次数  
- `retry_badcase_ratio_threshold`: float, 坏案例检测长度阈值  
- `use_random`: bool, 是否启用随机性  
- `output_path`: str, 输出wav路径  
- `verbose`: bool, 是否打印详细推理信息  

**示例 JSON**：

```json
{
  "params": {
    "text": "hello",
    "spk_audio_prompt": "examples/voice_01.wav",
    "emo_vector": [0,0,0,0,0,0,0,0],
    "cfg_value": 2,
    "inference_timesteps": 10,
    "normalize": false,
    "denoise": false,
    "retry_badcase": true,
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6,
    "use_random": false,
    "output_path": "tasks/output.wav",
    "verbose": true
  }
}
""",
)
async def submit_tts(req: TTSRequest):
    print(f"DEBUG: Received TTS request with params: {req.params}")
    task_id = str(uuid.uuid4())
    task_data = {
        "id": task_id,
        "status": "pending",
        "result": None,
        "worker_id": None,
        "submit_time": time.time(),
        "params": req.params,
    }
    (TASK_FOLDER / f"{task_id}.json").write_text(json.dumps(task_data))
    print(f"DEBUG: Created task {task_id}")
    return {"task_id": task_id}


@app.get(
    "/tts/{task_id}",
    summary="Get a specific TTS task by ID",
    description="""
Get details of a specific TTS task by its ID.

**Path Parameters**:

- `task_id`: str, The unique identifier of the task (UUID).

**Response**:

A task object containing:
- `id`: str, Unique task identifier (UUID).
- `status`: str, Current task status (`pending`, `running`, `done`, `error`).
- `result`: str or null, Path to the generated audio file (only present if status is `done`).
- `audio_data`: str or null, Base64-encoded audio data with data URI prefix (only present if status is `done` and audio file exists).
- `worker_id`: str or null, ID of the worker processing the task.
- `submit_time`: float, Timestamp when the task was submitted.
- `start_time`: float or null, Timestamp when processing started (only present if status is `running` or later).
- `end_time`: float or null, Timestamp when processing ended (only present if status is `done` or `error`).
- `duration`: float or null, Processing time in seconds.
- `params`: dict, TTS parameters used for the task.
- `error`: str or null, Error message (only present if status is `error`).

**Example Response** (status: done):

```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "done",
  "result": "tasks/123e4567-e89b-12d3-a456-426614174000.wav",
  "audio_data": "data:audio/wav;base64,UklGRnoGAABXQVZFZm10...",
  "worker_id": "worker-uuid",
  "submit_time": 1700000000.0,
  "start_time": 1700000001.0,
  "end_time": 1700000010.0,
  "duration": 9.0,
  "params": {"text": "Hello", "spk_audio_prompt": "examples/voice_01.wav"}
}
```
""",
)
def get_task(task_id: str):
    task_file = TASK_FOLDER / f"{task_id}.json"
    if not task_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    task_data = json.loads(task_file.read_text())
    if task_data.get("status") == "done" and task_data.get("result"):
        audio_path = Path(task_data["result"])
        if audio_path.exists():
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_data = (
                f"data:audio/wav;base64,{base64.b64encode(audio_bytes).decode('utf-8')}"
            )
            task_data["audio_data"] = audio_data
    return task_data


@app.get(
    "/tasks",
    response_model=TasksResponse,
    summary="Get TTS tasks filtered by status",
    description="""
Get a list of TTS tasks filtered by their status.

**Query Parameters**:

- `status`: str, Filter tasks by status. Options: `pending`, `running`, `done`, `error`. Default: `done`.

**Response**:

- `total`: int, Total number of tasks matching the filter.
- `tasks`: list, Array of task objects. Each task object contains:
  - `id`: str, Unique task identifier (UUID).
  - `status`: str, Current task status (`pending`, `running`, `done`, `error`).
  - `result`: str or null, Path to the generated audio file (only present if status is `done`).
  - `worker_id`: str or null, ID of the worker processing the task.
  - `submit_time`: float, Timestamp when the task was submitted.
  - `start_time`: float or null, Timestamp when processing started (only present if status is `running` or later).
  - `end_time`: float or null, Timestamp when processing ended (only present if status is `done` or `error`).
  - `duration`: float or null, Processing time in seconds.
  - `params`: dict, TTS parameters used for the task.
  - `error`: str or null, Error message (only present if status is `error`).

**Example Response**:

```json
{
  "total": 2,
  "tasks": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "status": "done",
      "result": "tasks/123e4567-e89b-12d3-a456-426614174000.wav",
      "worker_id": "worker-uuid",
      "submit_time": 1700000000.0,
      "params": {"text": "Hello", "spk_audio_prompt": "examples/voice_01.wav"}
    }
  ]
}
```
""",
)
def get_all_tasks(
    status: str = Query(
        "done", description="Filter tasks by status: pending, running, done, error"
    ),
):
    """Get list of TTS tasks filtered by status."""
    tasks = []
    for task_file in TASK_FOLDER.glob("*.json"):
        try:
            task_data = json.loads(task_file.read_text())
            if task_data.get("status") == status:
                tasks.append(task_data)
        except:
            continue
    return TasksResponse(total=len(tasks), tasks=tasks)


# --------------------
# __main__ 入口
# --------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",  # 注意这里用字符串：模块名:app
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开启自动重载
    )
