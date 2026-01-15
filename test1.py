import sys
import os
import time
import torch
from indextts.infer_v2 import IndexTTS2
# Using a separate, emotional reference audio file to condition the speech synthesis:
# --------------------
# Path fix
# --------------------
ROOT_DIR = "/content/tts"
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
    spk_audio_prompt='/content/tts/examples/voice_01.wav',
    text="hello",
    output_path="/content/warmup.wav",
    verbose=False
)
torch.cuda.synchronize()


# --------------------
# Benchmark
# --------------------
print("\n===== TTS Benchmark =====\n")


text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="/content/gen_test_1.wav", emo_audio_prompt="/content/tts/examples/emo_sad.wav", verbose=True)