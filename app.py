import os
import torch
import threading
import tempfile
import asyncio
import numpy as np
import gradio as gr
from huggingface_hub import login
from cached_path import cached_path
from vinorm import TTSnorm
import edge_tts
from pydub import AudioSegment

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

# Cấu hình Edge TTS chữa cháy
EDGE_VOICE = "vi-VN-HoaiMyNeural" # Giọng nữ | Nam dùng: vi-VN-NamMinhNeural
device_count = torch.cuda.device_count()
print(f"🚀 Nhím Review Engine: Phát hiện {device_count} GPU.")

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

# --- 3. LOGIC XỬ LÝ CHỮA CHÁY (EDGE TTS SSML) ---
async def generate_edge_tts_ssml(text, output_path):
    """Sử dụng SSML để tăng chất lượng câu ngắn, tránh nuốt chữ"""
    # Chỉnh pitch thấp xuống một chút (-5Hz) để gần với tông giọng clone hơn
    ssml_text = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="vi-VN">
        <voice name="{EDGE_VOICE}">
            <prosody pitch="-5Hz" rate="-5%">
                <break time="100ms" />
                {text}
                <break time="100ms" />
            </prosody>
        </voice>
    </speak>
    """
    communicate = edge_tts.Communicate(ssml_text, EDGE_VOICE)
    await communicate.save(output_path)

# --- 4. BỘ ĐIỀU PHỐI (LOAD BALANCER) ---
gpu_counter = 0
gpu_lock = threading.Lock()

def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ").replace(" .. ", " . ")
    text = text.replace('"', "")
    return " ".join(text.split())

def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):
    global gpu_counter
    if not ref_audio_orig or not gen_text.strip():
        raise gr.Error("Thiếu dữ liệu đầu vào.")

    # Luân phiên GPU
    with gpu_lock:
        worker_id = gpu_counter % len(models)
        gpu_counter += 1
    
    selected_model = models[worker_id]
    selected_vocoder = vocoders[worker_id]
    selected_device = devices[worker_id]

    # Chuẩn hóa văn bản trước
    norm_text = post_process(TTSnorm(gen_text)).lower()

    try:
        # KIỂM TRA TEXT TRỐNG (Fix lỗi Reshape Tensor)
        if not norm_text.strip():
            raise ValueError("Văn bản trống sau chuẩn hóa")

        # THỬ LẦN 1: Dùng F5-TTS (Ưu tiên giọng Clone)
        with torch.cuda.device(selected_device):
            # Với câu quá ngắn (< 3 từ), F5 thường lỗi, chủ động dùng Edge luôn
            if len(norm_text.split()) <= 3:
                raise ValueError("Câu quá ngắn, chuyển sang Edge TTS")

            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
            final_wave, final_sample_rate, spectrogram = infer_process(
                ref_audio, ref_text.lower(), norm_text, 
                selected_model, selected_vocoder, speed=speed
            )
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
                save_spectrogram(spectrogram, tmp_spectrogram.name)
                
            return (final_sample_rate, final_wave), tmp_spectrogram.name

    except Exception as e:
        # THỬ LẦN 2: CHỮA CHÁY BẰNG EDGE TTS
        print(f"⚠️ Worker {worker_id} lỗi: {e}. Đang dùng Edge TTS dự phòng...")
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_edge:
                asyncio.run(generate_edge_tts_ssml(gen_text, tmp_edge.name))
                edge_audio = AudioSegment.from_file(tmp_edge.name)
                
                # Chuyển đổi định dạng để Gradio hiểu (numpy)
                samples = np.array(edge_audio.get_array_of_samples()).astype(np.float32) / 32768.0
                if edge_audio.channels > 1:
                    samples = samples.reshape((-1, edge_audio.channels)).mean(axis=1)
                
                return (edge_audio.frame_rate, samples), None
        except Exception as edge_err:
            print(f"❌ Cả 2 engine đều thất bại: {edge_err}")
            # Trả về 0.5s im lặng để không sập luồng
            return (44100, np.zeros(int(44100 * 0.5), dtype=np.float32)), None

# --- 5. GIAO DIỆN GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Nhím Review - F5-TTS Dual GPU Engine")
    gr.Markdown(f"Server đang chạy trên **{len(models)} GPU T4**. Chế độ xử lý 20 luồng song song.")
    
    with gr.Row():
        ref_audio = gr.Audio(label="Giọng mẫu", type="filepath")
        gen_text = gr.Textbox(label="Văn bản", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="Tốc độ")
    btn = gr.Button("🔥 TẠO GIỌNG NÓI", variant="primary")
    
    with gr.Row():
        out_aud = gr.Audio(label="Kết quả", type="numpy")
        out_img = gr.Image(label="Spectrogram")

    btn.click(infer_tts, [ref_audio, gen_text, speed], [out_aud, out_img])

# Cấu hình hàng đợi để xử lý dồn 20 luồng
demo.queue(max_size=50).launch(share=True, max_threads=20)
