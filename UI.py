# UI.py
# 使用 Gradio 創建用户界面，集成 LLM、TTS 和視頻生成模組

import os
import gradio as gr
import tempfile
from llm_module import generate_text
from tts_module import text_to_speech, get_taiwanese_voices
from video_module import generate_video
import time

# 設置模型路徑
MODEL_PATHS = {
    "unet_model_path": "./models/musetalkV15/unet.pth",
    "unet_config": "./models/musetalkV15/musetalk.json",
    "whisper_dir": "./models/whisper"
}

# 可用的視頻模板
VIDEO_TEMPLATES = {
    "孫燕姿": "assets/demo/sun1/sun.png",
    "蒙娜麗莎": "assets/demo/monalisa/monalisa.png",
    "伊隆馬斯克": "assets/demo/musk/musk.png",
    "普通男性": "assets/demo/man/man.png"
}

# 可用的 TTS 聲音
TTS_VOICES = {
    "女性聲音 (曉臻)": "zh-TW-HsiaoChenNeural",
    "男性聲音 (雲哲)": "zh-TW-YunJheNeural",
    "女孩聲音 (曉玉)": "zh-TW-HsiaoYuNeural"
}

# 創建輸出目錄（如果不存在）
os.makedirs("./outputs", exist_ok=True)

def process_query(query, selected_template, selected_voice, progress=gr.Progress()):
    """
    處理用户查詢的完整流程：LLM → TTS → 視頻生成
    """
    results = {
        "llm_output": "",
        "audio_path": "",
        "video_path": ""
    }
    
    # 步驟 1: 使用 LLM 生成文本
    progress(0.1, "使用 LLM 生成回應中...")
    try:
        llm_output = generate_text(query)
        results["llm_output"] = llm_output
    except Exception as e:
        return {
            "llm_output": f"LLM 錯誤: {str(e)}",
            "audio_path": None,
            "video_path": None
        }
    
    # 步驟 2: 使用 TTS 生成音頻
    progress(0.3, "將文本轉換為語音中...")
    try:
        timestamp = int(time.time())
        audio_file = f"./outputs/audio_{timestamp}.mp3"
        audio_path = text_to_speech(
            text=llm_output,
            output_file=audio_file,
            voice=selected_voice
        )
        results["audio_path"] = audio_path
    except Exception as e:
        return {
            "llm_output": results["llm_output"],
            "audio_path": f"TTS 錯誤: {str(e)}",
            "video_path": None
        }
    
    # 步驟 3: 生成視頻
    progress(0.6, "生成嘴型同步視頻中...")
    try:
        video_file = f"./outputs/video_{timestamp}.mp4"
        video_template = VIDEO_TEMPLATES[selected_template]
        
        video_path = generate_video(
            audio_file=audio_path,
            video_path=video_template,
            bbox_shift=0,
            extra_margin=10,
            output_path=video_file,
            unet_model_path=MODEL_PATHS["unet_model_path"],
            unet_config=MODEL_PATHS["unet_config"],
            whisper_dir=MODEL_PATHS["whisper_dir"]
        )
        results["video_path"] = video_path
    except Exception as e:
        return {
            "llm_output": results["llm_output"],
            "audio_path": results["audio_path"],
            "video_path": f"視頻生成錯誤: {str(e)}"
        }
    
    progress(1.0, "處理完成!")
    return results

def build_interface():
    """
    構建 Gradio 界面
    """
    with gr.Blocks(title="MuseTalk 對話視頻生成器", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# MuseTalk 對話視頻生成器")
        gr.Markdown("輸入您的問題，系統將生成文本回應並創建對應的視頻")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 輸入區域
                query_input = gr.Textbox(
                    label="您的問題", 
                    placeholder="請輸入您想問的問題...",
                    lines=3
                )
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        choices=list(VIDEO_TEMPLATES.keys()),
                        value=list(VIDEO_TEMPLATES.keys())[0],
                        label="選擇視頻模板"
                    )
                    voice_dropdown = gr.Dropdown(
                        choices=list(TTS_VOICES.keys()),
                        value=list(TTS_VOICES.keys())[0],
                        label="選擇語音"
                    )
                submit_btn = gr.Button("生成視頻", variant="primary")
            
            with gr.Column(scale=3):
                # 輸出區域
                llm_output = gr.Textbox(label="AI 回應文本", lines=5)
                audio_output = gr.Audio(label="生成的音頻")
                video_output = gr.Video(label="生成的視頻")
        
        # 處理提交
        submit_btn.click(
            fn=process_query,
            inputs=[
                query_input,
                template_dropdown,
                gr.State(lambda x: TTS_VOICES[x])(voice_dropdown)
            ],
            outputs={
                "llm_output": llm_output,
                "audio_path": audio_output, 
                "video_path": video_output
            }
        )
        
        # 範例
        gr.Examples(
            examples=[
                ["介紹一下台灣的夜市文化", "孫燕姿", "女性聲音 (曉臻)"],
                ["請分享一些關於台北101的歷史", "伊隆馬斯克", "男性聲音 (雲哲)"],
                ["台灣有哪些著名的小吃？", "蒙娜麗莎", "女孩聲音 (曉玉)"]
            ],
            inputs=[query_input, template_dropdown, voice_dropdown]
        )
        
        gr.Markdown("## 使用說明")
        gr.Markdown("""
        1. 在文本框中輸入您想問的問題
        2. 選擇想要使用的視頻模板和語音類型
        3. 點擊「生成視頻」按鈕
        4. 等待系統生成文本、音頻和視頻
        5. 您可以下載生成的音頻和視頻
        """)
        
    return interface

# 啟動 Gradio 界面
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=True)