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
# 初始化模型并 warmup
# --------------------
tts_model = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False,
)
print(f"[Worker {WORKER_ID}] Model loaded. Warmup starting...")

# warmup
warmup_path = TASK_FOLDER / "warmup.wav"
tts_model.infer(
    spk_audio_prompt="examples/voice_01.wav",
    text="Warmup",
    output_path=str(warmup_path),
    verbose=True,
)
torch.cuda.synchronize()
print(f"[Worker {WORKER_ID}] Warmup done: {warmup_path}")


# --------------------
# 处理单个任务（带详细日志）
# --------------------
def run_indextts_task(task_file):
    try:
        task_data = json.loads(task_file.read_text())
    except Exception as e:
        print(f"[Worker {WORKER_ID}] [ERROR] Failed to read {task_file}: {e}")
        return

    if task_data.get("status") != "pending":
        print(
            f"[Worker {WORKER_ID}] Skipping task {task_data.get('id')} (status: {task_data.get('status')})"
        )
        return

    task_id = task_data["id"]
    task_params = task_data["params"]

    # 自动生成 output_path
    if "output_path" not in task_params:
        task_params["output_path"] = str(TASK_FOLDER / f"{task_id}.wav")

    # 如果缺少 spk_audio_prompt，自动补充
    if "spk_audio_prompt" not in task_params:
        task_params["spk_audio_prompt"] = "examples/voice_01.wav"
        print(
            f"[Worker {WORKER_ID}] Task {task_id} missing spk_audio_prompt, using default."
        )

    # 更新状态为 running
    task_data["status"] = "running"
    task_data["worker_id"] = WORKER_ID
    task_data["start_time"] = time.time()
    task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
    print(
        f"[Worker {WORKER_ID}] [RUNNING] Task {task_id} started at {task_data['start_time']:.2f}"
    )

    try:
        # 调用 TTS 生成音频
        print(f"[Worker {WORKER_ID}] [INFO] Generating audio for task {task_id}...")
        tts_model.infer(**task_params)
        torch.cuda.synchronize()

        # 更新状态为 done
        task_data["status"] = "done"
        task_data["result"] = task_params["output_path"]
        task_data["end_time"] = time.time()
        task_data["duration"] = task_data["end_time"] - task_data["start_time"]
        task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
        print(
            f"[Worker {WORKER_ID}] [DONE] Task {task_id} completed in {task_data['duration']:.2f}s, output: {task_params['output_path']}"
        )

    except Exception as e:
        # 更新状态为 error
        task_data["status"] = "error"
        task_data["error"] = str(e)
        task_data["end_time"] = time.time()
        task_data["duration"] = task_data["end_time"] - task_data.get(
            "start_time", task_data["end_time"]
        )
        task_file.write_text(json.dumps(task_data, ensure_ascii=False, indent=2))
        print(
            f"[Worker {WORKER_ID}] [ERROR] Task {task_id} failed after {task_data['duration']:.2f}s: {e}"
        )


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
