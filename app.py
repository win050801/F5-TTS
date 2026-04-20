import os
import pysrt
import time
import gradio as gr
from pydub import AudioSegment, effects
from gradio_client import Client, handle_file
import concurrent.futures

def time_to_ms(srt_time):
    return (srt_time.hours * 3600000 + srt_time.minutes * 60000 + 
            srt_time.seconds * 1000 + srt_time.milliseconds)

def get_tts_audio(client, text, ref_audio, speed, index):
    """Hàm phụ trợ gọi API cho từng câu đơn lẻ"""
    if not text.strip(): return None
    proc_text = text if len(text.split()) > 3 else text + " . . ."
    try:
        result = client.predict(
            ref_audio_orig=handle_file(ref_audio),
            gen_text=proc_text,
            speed=speed,
            api_name="/infer_tts"
        )
        gen_path = result[0] if isinstance(result, (list, tuple)) else result
        return AudioSegment.from_file(gen_path)
    except Exception as e:
        print(f"⚠️ Lỗi câu {index}: {e}")
        return None

def process_batch_pair(index_pair, pair_subs, api_link, ref_audio, speed):
    """
    Xử lý một cặp câu (2 câu) cùng lúc.
    Tính toán khoảng lặng chính xác giữa 2 câu.
    """
    client = Client(api_link)
    combined_audio = AudioSegment.empty()
    
    # Lấy câu 1
    sub1 = pair_subs[0]
    audio1 = get_tts_audio(client, sub1.text, ref_audio, speed, f"{index_pair}a")
    
    if len(pair_subs) > 1:
        # Lấy câu 2
        sub2 = pair_subs[1]
        audio2 = get_tts_audio(client, sub2.text, ref_audio, speed, f"{index_pair}b")
        
        # Tính toán Gap giữa câu 1 và câu 2
        gap_ms = time_to_ms(sub2.start) - time_to_ms(sub1.end)
        if gap_ms < 0: gap_ms = 0
        
        # Ghép nối: Audio1 + Im lặng (Gap) + Audio2
        silence = AudioSegment.silent(duration=gap_ms)
        
        # Fallback nếu audio lỗi
        a1 = audio1 if audio1 else AudioSegment.silent(duration=(time_to_ms(sub1.end)-time_to_ms(sub1.start)))
        a2 = audio2 if audio2 else AudioSegment.silent(duration=(time_to_ms(sub2.end)-time_to_ms(sub2.start)))
        
        combined_audio = a1 + silence + a2
    else:
        # Trường hợp câu lẻ cuối cùng
        combined_audio = audio1 if audio1 else AudioSegment.silent(duration=(time_to_ms(sub1.end)-time_to_ms(sub1.start)))

    # Trả về start_time của câu đầu tiên trong cặp để overlay chính xác
    return time_to_ms(sub1.start), combined_audio

def run_dubbing_max_gpu(api_link, ref_audio, srt_file, global_speed):
    try:
        subs = pysrt.open(srt_file.name, encoding='utf-8')
        # Gom các câu thành từng cặp [(sub1, sub2), (sub3, sub4), ...]
        pairs = [subs[i:i+2] for i in range(0, len(subs), 2)]
        total_pairs = len(pairs)
        results = []

        # TĂNG CỬA SỔ LÊN 10-15 ĐỂ ÉP GPU
        max_window = 10 
        
        print(f"🔥 Đang chạy {total_pairs} cặp câu với {max_window} luồng song song...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_window) as executor:
            futures = {}
            current_idx = 0

            # Gửi các cặp câu đầu tiên
            while current_idx < min(max_window, total_pairs):
                f = executor.submit(process_batch_pair, current_idx, pairs[current_idx], api_link, ref_audio, global_speed)
                futures[f] = current_idx
                current_idx += 1

            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    start_ms, audio_seg = f.result()
                    results.append((start_ms, audio_seg))
                    
                    del futures[f]
                    if current_idx < total_pairs:
                        new_f = executor.submit(process_batch_pair, current_idx, pairs[current_idx], api_link, ref_audio, global_speed)
                        futures[new_f] = current_idx
                        current_idx += 1
                    print(f"✅ Đã xong cặp {len(results)}/{total_pairs}")

        # Tổng hợp file
        print("🎵 Đang xuất file...")
        last_end_ms = time_to_ms(subs[-1].end) + 2000
        final_audio = AudioSegment.silent(duration=last_end_ms)
        
        for start_pos, seg in results:
            final_audio = final_audio.overlay(seg, position=start_pos)

        out_name = "nhim_review_max_gpu.wav"
        final_audio.export(out_name, format="wav")
        return "✅ Đã vắt kiệt GPU! File đã sẵn sàng.", out_name

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", None

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🚀 F5-TTS GPU BATCHER (Gom cặp + Gap chính xác)")
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="API URL")
            ref = gr.Audio(label="Giọng mẫu", type="filepath")
            srt = gr.File(label="File SRT")
            spd = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Speed")
            btn = gr.Button("🔥 CHẠY ÉP XUNG GPU", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="Status")
            out = gr.Audio(label="Kết quả")
    btn.click(run_dubbing_max_gpu, [url, ref, srt, spd], [status, out])

demo.launch()
