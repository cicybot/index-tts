import asyncio
import time
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("./tasks")
TASK_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="IndexTTS TTS API")

# --------------------
# 内存任务队列
# --------------------
task_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

from pydantic import BaseModel, Field
from typing import List, Optional


class TTSRequest(BaseModel):
    params: Dict[str, Any] = Field(
        ...,
        description=(
            "所有可控参数，直接传给 IndexTTS2.infer()\n\n"
            "可用字段示例:\n"
            " - text: str, 要合成的文本\n"
            " - spk_audio_prompt: str, 参考音频路径，用于克隆说话人\n"
            " - emo_vector: list[float], 8维情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]\n"
            " - cfg_value: float, LM引导强度，越高越贴合文本\n"
            " - inference_timesteps: int, LocDiT推理步数，越高效果越好\n"
            " - normalize: bool, 是否启用外部文本归一化\n"
            " - denoise: bool, 是否启用降噪\n"
            " - retry_badcase: bool, 是否重试坏案例\n"
            " - retry_badcase_max_times: int, 最大重试次数\n"
            " - retry_badcase_ratio_threshold: float, 坏案例检测长度阈值\n"
            " - use_random: bool, 是否启用随机性\n"
            " - output_path: str, 输出wav路径\n"
            " - verbose: bool, 是否打印详细推理信息\n\n"
            "示例JSON:\n"
            "{\n"
            '  "params": {\n'
            '    "text": "hello",\n'
            '    "spk_audio_prompt": "examples/voice_01.wav",\n'
            '    "emo_vector": [0,0,0,0,0,0,0,0],\n'
            '    "cfg_value": 2,\n'
            '    "inference_timesteps": 10,\n'
            '    "normalize": false,\n'
            '    "denoise": false,\n'
            '    "retry_badcase": true,\n'
            '    "retry_badcase_max_times": 3,\n'
            '    "retry_badcase_ratio_threshold": 6,\n'
            '    "use_random": false,\n'
            '    "output_path": "tasks/output.wav",\n'
            '    "verbose": true\n'
            '  }\n'
            "}"
        )
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
    uvicorn.run(
        "api:app",      # 注意这里用字符串：模块名:app
        host="0.0.0.0",
        port=8000,
        reload=True,    # 开启自动重载
     )
