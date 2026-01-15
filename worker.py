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
# 初始化模型（只加载一次）
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
# 异步 worker 循环（扫描本地 JSON 任务）
# --------------------
async def worker_loop(poll_interval=1.0):
    print(f"[Worker {WORKER_ID}] Worker loop started")
    while True:
        # 查找所有 pending 的任务
        for task_file in TASK_FOLDER.glob("*.json"):
            try:
                task_data = json.loads(task_file.read_text())
            except Exception as e:
                print(f"[Worker {WORKER_ID}] Failed to read {task_file}: {e}")
                continue

            if task_data.get("status") != "pending":
                continue  # 已经在处理中或完成

            task_id = task_data["id"]
            task_params = task_data["params"]

            # 自动生成 output_path
            if "output_path" not in task_params:
                task_params["output_path"] = str(TASK_FOLDER / f"{task_id}.wav")

            task_data["status"] = "running"
            task_data["worker_id"] = WORKER_ID
            task_data["start_time"] = time.time()
            task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))

            print(f"[Worker {WORKER_ID}] Start task {task_id}")

            try:
                await asyncio.to_thread(tts_model.infer, **task_params)

                task_data["status"] = "done"
                task_data["result"] = task_params["output_path"]
                task_data["end_time"] = time.time()
                task_data["duration"] = task_data["end_time"] - task_data["start_time"]
                task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
                print(f"[Worker {WORKER_ID}] Task {task_id} done in {task_data['duration']:.2f}s")

            except Exception as e:
                task_data["status"] = "error"
                task_data["error"] = str(e)
                task_data["end_time"] = time.time()
                task_data["duration"] = task_data["end_time"] - task_data.get("start_time", task_data["end_time"])
                task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
                print(f"[Worker {WORKER_ID}] Task {task_id} error: {e}")

        await asyncio.sleep(poll_interval)  # 等待一段时间再扫描

# --------------------
# 启动 worker
# --------------------
if __name__ == "__main__":
    asyncio.run(worker_loop())
