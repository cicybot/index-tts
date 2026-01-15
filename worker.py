import asyncio
import uuid
import json
from pathlib import Path
from indextts.infer_v2 import IndexTTS2
import torch
import time

# --------------------
# 任务存储路径
# --------------------
TASK_FOLDER = Path("./tasks")
TASK_FOLDER.mkdir(exist_ok=True)

# --------------------
# Worker ID
# --------------------
WORKER_ID = str(uuid.uuid4())
print(f"[worker {WORKER_ID}] Starting worker...")

# --------------------
# 初始化模型（只加载一次）
# --------------------
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False
)
print(f"[worker {WORKER_ID}] Model loaded.")

# --------------------
# Warmup
# --------------------
print(f"[worker {WORKER_ID}] Warming up model...")
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="Warmup",
    output_path=str(TASK_FOLDER / "warmup.wav"),
    verbose=False
)
torch.cuda.synchronize()
print(f"[worker {WORKER_ID}] Warmup done.")

# --------------------
# 异步 worker 循环
# --------------------
async def worker_loop(task_queue: asyncio.Queue):
    while True:
        try:
            task_id = await task_queue.get()
            task_file = TASK_FOLDER / f"{task_id}.json"
            if not task_file.exists():
                print(f"[worker {WORKER_ID}] Task {task_id} file missing")
                task_queue.task_done()
                continue

            # 读取任务
            task_data = json.loads(task_file.read_text())

            # 记录执行开始时间
            task_data["status"] = "running"
            task_data["worker_id"] = WORKER_ID
            task_data["start_time"] = time.time()
            task_file.write_text(json.dumps(task_data))

            # --------------------
            # 执行 TTS
            # --------------------
            text = task_data.get("text", "Hello")
            spk_prompt = task_data.get("spk_audio_prompt", None)
            emo_vector = task_data.get("emo_vector", None)
            use_random = task_data.get("use_random", False)

            # 输出文件路径
            output_path = TASK_FOLDER / f"{task_id}.wav"

            # 执行推理
            wav = await asyncio.to_thread(
                tts.infer,
                spk_audio_prompt=spk_prompt,
                text=text,
                output_path=str(output_path),
                emo_vector=emo_vector,
                use_random=use_random,
                verbose=True
            )

            # --------------------
            # 更新任务状态
            # --------------------
            task_data["status"] = "done"
            task_data["result"] = str(output_path)
            task_data["end_time"] = time.time()
            # 计算执行耗时
            task_data["duration"] = task_data["end_time"] - task_data["start_time"]
            task_file.write_text(json.dumps(task_data))
            print(f"[worker {WORKER_ID}] Task {task_id} done in {task_data['duration']:.2f}s")

        except Exception as e:
            print(f"[worker {WORKER_ID}] Error: {e}")
            if 'task_data' in locals():
                task_data["status"] = "error"
                task_data["error"] = str(e)
                task_data["end_time"] = time.time()
                task_data["duration"] = task_data["end_time"] - task_data.get("start_time", task_data["end_time"])
                task_file.write_text(json.dumps(task_data))
        finally:
            task_queue.task_done()
            await asyncio.sleep(0.01)


# --------------------
# 示例 main，用内存队列模拟
# --------------------
async def main():
    task_queue = asyncio.Queue(maxsize=16)

    # 测试任务入队
    sample_task_id = str(uuid.uuid4())
    sample_task = {
        "id": sample_task_id,
        "text": "哇塞！这个爆率也太高了！欧皇附体了！",
        "spk_audio_prompt": "examples/voice_07.wav",
        "emo_vector": [0, 0, 0, 0, 0, 0, 0.45, 0],
        "use_random": False,
        "status": "pending",
        "result": None,
        "submit_time": time.time()  # 提交时间
    }
    (TASK_FOLDER / f"{sample_task_id}.json").write_text(json.dumps(sample_task))
    await task_queue.put(sample_task_id)

    # 启动 worker 循环
    await worker_loop(task_queue)


if __name__ == "__main__":
    asyncio.run(main())
