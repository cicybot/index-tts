import asyncio
import json
import time
import uuid
from pathlib import Path

import torch
from indextts.infer_v2 import IndexTTS2

# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("./tasks")
TASK_FOLDER.mkdir(parents=True, exist_ok=True)

# --------------------
# Worker ID
# --------------------
WORKER_ID = str(uuid.uuid4())
print(f"[Worker {WORKER_ID}] Starting worker...")

# --------------------
# 初始化模型（只加载一次）并 warmup
# --------------------
tts_model = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False
)
print(f"[Worker {WORKER_ID}] Model loaded. Warmup starting...")

# warmup
tts_model.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="Warmup",
    output_path=str(TASK_FOLDER / "warmup.wav"),
    verbose=False
)
torch.cuda.synchronize()
print(f"[Worker {WORKER_ID}] Warmup done.")

# --------------------
# 处理单个任务
# --------------------
def run_indextts_task(task_file):
    try:
        task_data = json.loads(task_file.read_text())
    except Exception as e:
        print(f"[Worker {WORKER_ID}] Failed to read {task_file}: {e}")
        return

    if task_data.get("status") != "pending":
        return

    task_id = task_data["id"]
    task_params = task_data["params"]

    # 自动生成 output_path
    if "output_path" not in task_params:
        task_params["output_path"] = str(TASK_FOLDER / f"{task_id}.wav")

    # 更新状态为 running
    task_data["status"] = "running"
    task_data["worker_id"] = WORKER_ID
    task_data["start_time"] = time.time()
    task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
    print(f"[Worker {WORKER_ID}] Start task {task_id}")

    try:
        # 调用 TTS 生成音频
        tts_model.infer(**task_params)
        torch.cuda.synchronize()

        # 更新状态为 done
        task_data["status"] = "done"
        task_data["result"] = task_params["output_path"]
        task_data["end_time"] = time.time()
        task_data["duration"] = task_data["end_time"] - task_data["start_time"]
        task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
        print(f"[Worker {WORKER_ID}] Task {task_id} done in {task_data['duration']:.2f}s")

    except Exception as e:
        # 更新状态为 error
        task_data["status"] = "error"
        task_data["error"] = str(e)
        task_data["end_time"] = time.time()
        task_data["duration"] = task_data["end_time"] - task_data.get("start_time", task_data["end_time"])
        task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
        print(f"[Worker {WORKER_ID}] Task {task_id} error: {e}")


# --------------------
# 无限循环扫描任务
# --------------------
def run_task(poll_interval=1.0):
    print(f"[Worker {WORKER_ID}] Worker loop started")
    while True:
        for task_file in TASK_FOLDER.glob("*.json"):
            run_indextts_task(task_file)

        time.sleep(poll_interval)


# --------------------
# 启动
# --------------------
if __name__ == "__main__":
    run_task()
