import sys
import os
import time
import torch
from indextts.infer_v2 import IndexTTS2

# --------------------
# Path fix
# --------------------
ROOT_DIR = "/content/index-tts"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------
# CUDA check
# --------------------
assert torch.cuda.is_available(), "❌ CUDA 不可用，请在 Colab 开启 GPU"

# --------------------
# Model init (❗不计时)
# --------------------
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False
)

# --------------------
# Warmup (非常重要)
# --------------------
tts.infer(
    spk_audio_prompt='/content/data/assets/voice_01.wav',
    text="hello",
    output_path="/content/warmup.wav",
    verbose=False
)
torch.cuda.synchronize()

# --------------------
# Test cases
# --------------------
test_list = [
    "There is no wine in this country, the young man said.",
    "Translate for me, what is a surprise!"
]

# --------------------
# Benchmark
# --------------------
print("\n===== TTS Benchmark =====\n")

for i, text in enumerate(test_list, 1):
    torch.cuda.synchronize()
    start = time.perf_counter()

    tts.infer(
        spk_audio_prompt='/content/data/assets/voice_01.wav',
        text=text,
        output_path=f"/content/gen_{i}.wav",
        verbose=False
    )

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"[{i}] Text: {text}")
    print(f"    ⏱️ Inference time: {end - start:.3f} seconds\n")
