import os
import torch
import threading
import tempfile
import asyncio
import numpy as np
import gradio as gr
import uuid
from huggingface_hub import login
from cached_path import cached_path
from vinorm import TTSnorm
import edge_tts
from pydub import AudioSegment

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text, load_vocoder, load_model,
    infer_process, save_spectrogram,
)

# 1. Cấu hình & Login
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token: login(token=hf_token)

EDGE_VOICE = "vi-VN-HoaiMyNeural" 
device_count = torch.cuda.device_count()

# 2. Nạp Model song song
def load_vivoice_model(device):
    model = load_model(
        DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
        vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
    )
    return model.to(device)

models, vocoders, devices = [], [], []
for i in range(min(device_count, 2)):
    dev = f"cuda:{i}"
    models.append(load_vivoice_model(dev)); vocoders.append(load_vocoder().to(dev)); devices.append(dev)

# 3. Edge-TTS SSML (Chữa cháy chuyên nghiệp)
async def generate_edge_tts_safe(text, output_path, speed_factor=1.0):
    rate = f"{int((speed_factor - 1) * 100)}%"
    if speed_factor >= 1: rate = "+" + rate
    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="vi-VN">
        <voice name="{EDGE_VOICE}"><prosody pitch="-5Hz" rate="{rate}" volume="+20%">
        <break time="150ms" />{text}<break time="100ms" /></prosody></voice></speak>"""
    await edge_tts.Communicate(ssml, EDGE_VOICE).save(output_path)

# 4. Điều phối GPU
gpu_counter = 0
gpu_lock = threading.Lock()

def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0):
    global gpu_counter
    with gpu_lock:
        worker_id = gpu_counter % len(models)
        gpu_counter += 1
    
    selected_model, selected_vocoder, selected_device = models[worker_id], vocoders[worker_id], devices[worker_id]
    request_id = str(uuid.uuid4())
    temp_edge = f"edge_{request_id}.mp3"
    norm_text = " ".join(TTSnorm(gen_text).replace('"', "").split()).lower()

    try:
        if not norm_text.strip(): raise ValueError("Rỗng")
        with torch.cuda.device(selected_device):
            if len(norm_text.split()) <= 3: raise ValueError("Câu ngắn")
            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
            wave, sr, spec = infer_process(ref_audio, ref_text.lower(), norm_text, selected_model, selected_vocoder, speed=speed)
            return (sr, wave), None
    except Exception:
        try:
            asyncio.run(generate_edge_tts_safe(norm_text, temp_edge, speed))
            audio = AudioSegment.from_file(temp_edge)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            if audio.channels > 1: samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            if os.path.exists(temp_edge): os.remove(temp_edge)
            return (audio.frame_rate, samples), None
        except:
            return (44100, np.zeros(int(44100 * 0.5), dtype=np.float32)), None

# 5. Giao diện & Hàng đợi 20 luồng
demo = gr.Interface(fn=infer_tts, inputs=[gr.Audio(type="filepath"), gr.Textbox(), gr.Slider(0.3, 2.0, 1.0)], outputs=[gr.Audio(type="numpy"), gr.Image()])
demo.queue(max_size=50).launch(share=True, max_threads=20)
