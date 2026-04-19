import os
import torch
import threading
import tempfile
import gradio as gr
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

# 1. Cấu hình thiết bị và Token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)

device_count = torch.cuda.device_count()
print(f"🚀 Phát hiện {device_count} GPU. Đang khởi tạo...")

# 2. Tải Model lên 2 GPU độc lập
def load_vivoice_model(device):
    print(f"📦 Đang nạp Model lên {device}...")
    model = load_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
        vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
    )
    return model.to(device)

models = []
vocoders = []
devices = []

# Nạp model vào danh sách ứng với từng GPU
for i in range(min(device_count, 2)):
    dev = f"cuda:{i}"
    models.append(load_vivoice_model(dev))
    vocoders.append(load_vocoder().to(dev))
    devices.append(dev)

# Dự phòng nếu không có GPU
if not models:
    dev = "cpu"
    models.append(load_vivoice_model(dev))
    vocoders.append(load_vocoder().to(dev))
    devices.append(dev)

# 3. Điều phối luồng và GPU
gpu_counter = 0
gpu_lock = threading.Lock()

def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ").replace(" .. ", " . ")
    text = text.replace(" , , ", " , ").replace(" ,, ", " , ")
    text = text.replace('"', "")
    return " ".join(text.split())

def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):
    global gpu_counter
    
    if not ref_audio_orig or not gen_text.strip():
        raise gr.Error("Vui lòng cung cấp đủ giọng mẫu và văn bản.")

    # Chọn worker (GPU) theo kiểu xoay vòng
    with gpu_lock:
        worker_id = gpu_counter % len(models)
        gpu_counter += 1
    
    selected_model = models[worker_id]
    selected_vocoder = vocoders[worker_id]
    selected_device = devices[worker_id]
    
    print(f"🎙️ Worker {worker_id} đang xử lý trên {selected_device}")

    try:
        # QUAN TRỌNG: Ép toàn bộ tiến trình này phải chạy trên GPU đã chọn
        # Điều này sửa lỗi "Expected all tensors to be on the same device"
        with torch.cuda.device(selected_device):
            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
            
            # Chuẩn hóa văn bản
            norm_text = post_process(TTSnorm(gen_text)).lower()
            
            final_wave, final_sample_rate, spectrogram = infer_process(
                ref_audio, 
                ref_text.lower(), 
                norm_text, 
                selected_model, 
                selected_vocoder, 
                speed=speed
            )
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
                spectrogram_path = tmp_spectrogram.name
                save_spectrogram(spectrogram, spectrogram_path)

            return (final_sample_rate, final_wave), spectrogram_path
            
    except Exception as e:
        print(f"❌ Lỗi thực thi trên {selected_device}: {e}")
        raise gr.Error(f"Lỗi tạo giọng nói: {e}")

# 4. Giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🎤 F5-TTS Dual-GPU (Fixed)\nHệ thống đang dùng **{len(models)} GPU** để xử lý song song.")
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Giọng mẫu", type="filepath")
        gen_text = gr.Textbox(label="📝 Văn bản", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
    btn_synthesize = gr.Button("🔥 Tạo giọng nói", variant="primary")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Kết quả", type="numpy")
        output_spectrogram = gr.Image(label="📊 Biểu đồ phổ")

    btn_synthesize.click(
        fn=infer_tts, 
        inputs=[ref_audio, gen_text, speed], 
        outputs=[output_audio, output_spectrogram],
        concurrency_limit=len(models) # Xử lý song song tối đa bằng số GPU
    )

# Launch với hàng đợi tối ưu
demo.queue(default_concurrency_limit=len(models)).launch(share=True)
