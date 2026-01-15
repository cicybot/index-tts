import json
import time
import uuid
from pathlib import Path

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
# run_task 方法（无限循环，只打印任务）
# --------------------
def run_task(poll_interval=1.0):
    print(f"[Worker {WORKER_ID}] Worker loop started")
    while True:
        for task_file in TASK_FOLDER.glob("*.json"):
            try:
                task_data = json.loads(task_file.read_text())
            except Exception as e:
                print(f"[Worker {WORKER_ID}] Failed to read {task_file}: {e}")
                continue

            print(f"[Worker {WORKER_ID}] Found task {task_data.get('id', 'UNKNOWN')}:")
            print(json.dumps(task_data, ensure_ascii=False, indent=2))

        time.sleep(poll_interval)  # 等待一段时间再扫描

# --------------------
# 启动 run_task
# --------------------
if __name__ == "__main__":
    run_task()
