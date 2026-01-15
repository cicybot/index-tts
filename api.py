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

class TTSParams(BaseModel):
    # --- 必填 ---
    text: str = Field(..., description="待生成文本，必填")

    # --- 可选 ---
    spk_audio_prompt: Optional[str] = Field(None, description="示例音频路径，用于声音克隆")
    prompt_text: Optional[str] = Field(None, description="参考文本，可选")
    emo_vector: Optional[List[float]] = Field(
        None,
        description="情绪向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm], 长度 8"
    )
    cfg_value: Optional[float] = Field(2.0, description="LM guidance，文本约束强度")
    inference_timesteps: Optional[int] = Field(10, description="LocDiT 推理步数，越高越精细")
    normalize: Optional[bool] = Field(False, description="是否启用外部 TN 工具")
    denoise: Optional[bool] = Field(False, description="是否启用外部 Denoise 工具")
    retry_badcase: Optional[bool] = Field(True, description="是否开启自动重试")
    retry_badcase_max_times: Optional[int] = Field(3, description="最大重试次数")
    retry_badcase_ratio_threshold: Optional[float] = Field(6.0, description="重试检测阈值")
    use_random: Optional[bool] = Field(False, description="是否随机化生成")
    output_path: Optional[str] = Field(None, description="输出 wav 文件路径，不填自动生成")
    verbose: Optional[bool] = Field(False, description="是否打印推理日志")

class TTSRequest(BaseModel):
    """
    TTS 请求体模型
    params 字典会直接传给 IndexTTS2.infer。
    前端可以根据需求传入任意 infer 支持的参数。
    """
    params: TTSParams



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
