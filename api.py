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
class TTSRequest(BaseModel):
    """
    TTS 请求体模型
    params 字典会直接传给 IndexTTS2.infer。
    前端可以根据需求传入任意 infer 支持的参数。
    """

    params: Dict[str, Any] = Field(
        ...,
        description="""
可控参数示例：
{
    "text": "待生成的文本，必填",
    "spk_audio_prompt": "示例音频路径，可选，用于声音克隆",
    "prompt_text": "参考文本，可选",
    "emo_vector": [0,0,0,0,0,0,0.5,0],  # 情绪向量，顺序: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    "cfg_value": 2.0,       # LM guidance，数值越大，文本约束越强
    "inference_timesteps": 10,  # LocDiT 推理步数，越高效果越好但慢
    "normalize": false,     # 是否启用外部 TN 工具
    "denoise": false,       # 是否启用外部 denoise 工具
    "retry_badcase": true,  # 是否开启自动重试
    "retry_badcase_max_times": 3,
    "retry_badcase_ratio_threshold": 6.0,
    "use_random": false,    # 是否随机化生成
    "output_path": "./tasks/xxx.wav",  # 输出文件路径，可选，不填自动生成
    "verbose": true         # 是否打印推理日志
}
注意事项：
1. text 是必填字段。
2. output_path 可以不填，worker 会自动生成。
3. emo_vector 长度必须为 8。
4. 其他参数可选，根据 IndexTTS2.infer 支持的参数传入。
"""
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
    uvicorn.run(app,
                host="0.0.0.0",
                port=8000,
                reload=True,    # 开启自动重载
                workers=1       # 可选，多进程可提高并发，但开发时通常保持 1
     )
