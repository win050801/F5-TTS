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

# Kiểm tra GPU
device_count = torch.cuda.device_count()
print(f"🚀 Hệ thống phát hiện {device_count} GPU.")

# 2. Tải Model lên cả 2 GPU
def load_vivoice_model(device):
    print(f"📦 Đang nạp Model lên {device}...")
    model = load_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/model_last.pt")),
        vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-ViVoice/config.json")),
    )
    return model.to(device)

# Khởi tạo 2 bộ công cụ trên 2 GPU
models = []
vocoders = []
devices = []

if device_count >= 2:
    # Nạp cho GPU 0
    models.append(load_vivoice_model("cuda:0"))
    vocoders.append(load_vocoder().to("cuda:0"))
    devices.append("cuda:0")
    # Nạp cho GPU 1
    models.append(load_vivoice_model("cuda:1"))
    vocoders.append(load_vocoder().to("cuda:1"))
    devices.append("cuda:1")
else:
    # Nếu chỉ có 1 GPU hoặc CPU
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    models.append(load_vivoice_model(dev))
    vocoders.append(load_vocoder().to(dev))
    devices.append(dev)

# 3. Bộ điều phối yêu cầu (Load Balancer)
gpu_counter = 0
gpu_lock = threading.Lock()

def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace(" , , ", " , ")
    text = text.replace(" ,, ", " , ")
    text = text.replace('"', "")
    return " ".join(text.split())

# Hàm xử lý chính (Không dùng @spaces.GPU)
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0):
    global gpu_counter
    
    if not ref_audio_orig or not gen_text.strip():
        raise gr.Error("Vui lòng cung cấp đủ giọng mẫu và văn bản.")

    # Chọn GPU luân phiên
    with gpu_lock:
        worker_id = gpu_counter % len(models)
        gpu_counter += 1
    
    selected_model = models[worker_id]
    selected_vocoder = vocoders[worker_id]
    selected_device = devices[worker_id]
    
    print(f"🎙️ Đang xử lý trên {selected_device} (Yêu cầu số {gpu_counter})")

    try:
        # Tiền xử lý
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Chạy Inference trên GPU đã chọn
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            post_process(TTSnorm(gen_text)).lower(), 
            selected_model, 
            selected_vocoder, 
            speed=speed
        )
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        print(f"❌ Lỗi trên {selected_device}: {e}")
        raise gr.Error(f"Lỗi tạo giọng nói: {e}")

# 4. Giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="F5-TTS 2-GPU Pro") as demo:
    gr.Markdown("# 🎤 F5-TTS Vietnamese: Chế độ 2-GPU Ultra Speed")
    gr.Markdown(f"Hệ thống đang chạy trên **{len(models)} GPU** để tối ưu hóa hàng đợi.")
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Giọng mẫu", type="filepath")
        gen_text = gr.Textbox(label="📝 Văn bản", placeholder="Nhập nội dung...", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
    btn_synthesize = gr.Button("🔥 BẮT ĐẦU TẠO GIỌNG", variant="primary")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Kết quả", type="numpy")
        output_spectrogram = gr.Image(label="📊 Biểu đồ phổ")

    # Thiết lập xử lý song song cho nút bấm
    btn_synthesize.click(
        fn=infer_tts, 
        inputs=[ref_audio, gen_text, speed], 
        outputs=[output_audio, output_spectrogram],
        concurrency_limit=len(models) # Giới hạn số câu xử lý cùng lúc bằng số GPU
    )

# 5. Cấu hình Hàng đợi và Launch
# max_size=20 để khớp với yêu cầu 20 luồng của bạn
demo.queue(default_concurrency_limit=len(models)).launch(share=True)
