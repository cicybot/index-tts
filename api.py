import asyncio
import time
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional,Dict

# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("./tasks")
TASK_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------
# 静态文件目录（供 /media 使用）
# --------------------
MEDIA_FOLDER = Path("./media")
MEDIA_FOLDER.mkdir(parents=True, exist_ok=True)
# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="IndexTTS TTS API")



class TTSRequest(BaseModel):
    params: Dict[str, Any]



# --------------------
# 路由: / 重定向到 /docs
# --------------------
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")

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
"""
)
async def submit_tts(req: TTSRequest):

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
