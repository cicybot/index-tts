import sys
import os
import time
import torch
from indextts.infer_v2 import IndexTTS2

"""
It's also possible to omit the emotional reference audio and instead provide
 an 8-float list specifying the intensity of each emotion, 
 in the following order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]. 
 You can additionally use the use_random parameter to introduce stochasticity during inference; 
 the default is False, and setting it to True enables randomness:

"""
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


tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
text = "哇塞！这个爆率也太高了！欧皇附体了！"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="/content/gen_test3.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)