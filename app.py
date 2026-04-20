import os
import torch
import threading
import tempfile
import numpy as np
import gradio as gr
import uuid
from huggingface_hub import login
from cached_path import cached_path
from vinorm import TTSnorm

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
)

# --- 1. CẤU HÌNH HỆ THỐNG ---
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)

device_count = torch.cuda.device_count()
print(f"🚀 Nhím Review Engine: Chế độ thuần F5-TTS trên {device_count} GPU.")

# --- 2. NẠP MODEL LÊN 2 GPU ---
def load_vivoice_model(device):
    print(f"📦 Đang nạp F5-TTS lên {device}...")
    model = load_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
        vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
    )
    return model.to(device)

models, vocoders, devices = [], [], []
for i in range(min(device_count, 2)):
    dev = f"cuda:{i}"
    models.append(load_vivoice_model(dev))
    vocoders.append(load_vocoder().to(dev))
    devices.append(dev)

# --- 3. ĐIỀU PHỐI VÀ XỬ LÝ ---
gpu_counter = 0
gpu_lock = threading.Lock()

def post_process(text):
    text = text.replace(" . . ", " . ").replace(" .. ", " . ").replace('"', "")
    return " ".join(text.split())

def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):
    global gpu_counter
    
    # 1. Chuẩn hóa văn bản trước
    norm_text = post_process(TTSnorm(gen_text)).lower()
    word_count = len(norm_text.split())

    # --- THUẬT TOÁN TRẢ VỀ KHOẢNG LẶNG CHO CÂU NGẮN (ĐÃ SỬA) ---
    if word_count <= 3:
        print(f"⚠️ Câu quá ngắn ({word_count} từ), trả về khoảng lặng để tránh lỗi AI.")
        # Tính toán thời gian im lặng: 1 từ tương đương ~0.5s, tối đa 1.5s
        silence_sec = 0.5 if word_count == 0 else word_count * 0.5
        return (44100, np.zeros(int(44100 * silence_sec), dtype=np.float32)), None

    # 2. Điều phối GPU cho các câu bình thường
    with gpu_lock:
        worker_id = gpu_counter % len(models)
        gpu_counter += 1
    
    selected_model = models[worker_id]
    selected_vocoder = vocoders[worker_id]
    selected_device = devices[worker_id]
    
    try:
        with torch.cuda.device(selected_device):
            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
            
            final_wave, final_sample_rate, spectrogram = infer_process(
                ref_audio, 
                ref_text.lower(), 
                norm_text, 
                selected_model, 
                selected_vocoder, 
                speed=speed
            )
            
            # Giới hạn ảo giác âm thanh
            actual_sec = len(final_wave) / final_sample_rate
            if actual_sec > 25 and word_count < 10:
                print(f"⚠️ Phát hiện ảo giác ({actual_sec:.1f}s), tự động cắt bỏ.")
                final_wave = final_wave[:int(final_sample_rate * 5)]

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                save_spectrogram(spectrogram, tmp_img.name)
                
            return (final_sample_rate, final_wave), tmp_img.name

    except Exception as e:
        print(f"❌ Lỗi thực thi trên {selected_device}: {e}")
        # Fallback im lặng nếu có lỗi bất ngờ
        return (44100, np.zeros(int(44100 * 1.0), dtype=np.float32)), None

# --- 4. GIAO DIỆN ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Nhím Review - F5-TTS Pure Engine")
    gr.Markdown(f"Chế độ: **Tự động trả khoảng lặng cho câu dưới 3 từ** để tránh lỗi voice.")
    
    with gr.Row():
        ref_audio = gr.Audio(label="Giọng mẫu", type="filepath")
        gen_text = gr.Textbox(label="Văn bản", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Tốc độ")
    btn = gr.Button("🚀 TẠO GIỌNG CLONE", variant="primary")
    
    with gr.Row():
        out_aud = gr.Audio(label="Kết quả", type="numpy")
        out_img = gr.Image(label="Spectrogram")

    btn.click(infer_tts, [ref_audio, gen_text, speed], [out_aud, out_img])

demo.queue(max_size=50).launch(share=True, max_threads=20)
