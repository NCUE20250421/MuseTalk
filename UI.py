# UI.py
# 使用 Gradio 創建用户界面，集成 LLM、TTS 和視頻生成模組

import os
import gradio as gr
import tempfile
from llm_module import generate_text
from tts_module import text_to_speech, get_taiwanese_voices
from video_module import generate_video, preload_models
import time
import datetime

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

# 格式化時間函數
def format_time(seconds):
    """將秒數格式化為易讀的時間格式"""
    return str(datetime.timedelta(seconds=round(seconds)))

# 預載入 MuseTalk 模型
print("正在預先載入 MuseTalk 模型，請稍候...")
start_time = time.time()
models_loaded = preload_models(
    unet_model_path=MODEL_PATHS["unet_model_path"],
    unet_config=MODEL_PATHS["unet_config"],
    whisper_dir=MODEL_PATHS["whisper_dir"]
)
model_load_time = time.time() - start_time
print(f"MuseTalk 模型預載狀態: {'成功' if models_loaded else '失敗'}, 耗時: {format_time(model_load_time)}")

# 生成歡迎語音和視頻
def generate_welcome_assets():
    """
    生成歡迎語音和視頻，用於系統預熱
    """
    print("正在生成歡迎語音和視頻...")
    welcome_text = "歡迎使用數位司儀系統"
    welcome_audio_path = "./outputs/welcome_audio.mp3"
    welcome_video_path = "./outputs/welcome_video.mp4"
    
    total_start_time = time.time()
    
    # 生成歡迎語音
    try:
        tts_start_time = time.time()
        audio_path = text_to_speech(
            text=welcome_text,
            output_file=welcome_audio_path,
            voice="zh-TW-HsiaoChenNeural"  # 使用女性聲音
        )
        tts_time = time.time() - tts_start_time
        print(f"歡迎語音生成成功: {audio_path}, 耗時: {format_time(tts_time)}")
        
        # 如果模型已預載，生成歡迎視頻
        if models_loaded:
            # 使用孫燕姿模板生成歡迎視頻
            video_start_time = time.time()
            video_template = VIDEO_TEMPLATES["孫燕姿"]
            video_path = generate_video(
                audio_file=audio_path,
                video_path=video_template,
                output_path=welcome_video_path,
                use_preloaded_models=True  # 使用預先載入的模型
            )
            video_time = time.time() - video_start_time
            total_time = time.time() - total_start_time
            
            print(f"歡迎視頻生成成功: {video_path}, 耗時: {format_time(video_time)}")
            print(f"歡迎資源生成總耗時: {format_time(total_time)}")
            return welcome_text, audio_path, video_path, tts_time, video_time
        
        total_time = time.time() - total_start_time
        print(f"歡迎資源生成總耗時: {format_time(total_time)}")
        return welcome_text, audio_path, None, tts_time, 0
    except Exception as e:
        print(f"歡迎資源生成失敗: {e}")
        return "歡迎使用數位司儀系統", None, None, 0, 0

# 生成歡迎資源
welcome_text, welcome_audio, welcome_video, welcome_tts_time, welcome_video_time = generate_welcome_assets()

def process_query(query, selected_template, selected_voice, progress=gr.Progress()):
    """
    處理用户查詢的完整流程：LLM → TTS → 視頻生成
    """
    # 獲取實際的 TTS 語音 ID
    voice_id = TTS_VOICES[selected_voice]
    
    # 記錄總處理開始時間
    total_start_time = time.time()
    
    # 步驟 1: 使用 LLM 生成文本
    progress(0.1, "使用 LLM 生成回應中...")
    llm_start_time = time.time()
    try:
        llm_output = generate_text(query)
        llm_time = time.time() - llm_start_time
        llm_output_with_time = f"{llm_output}\n\n[LLM 耗時: {format_time(llm_time)}]"
    except Exception as e:
        llm_time = time.time() - llm_start_time
        error_msg = f"LLM 錯誤: {str(e)}\n[耗時: {format_time(llm_time)}]"
        return error_msg, None, None, {"llm": llm_time, "tts": 0, "video": 0, "total": llm_time}
    
    # 步驟 2: 使用 TTS 生成音頻
    progress(0.3, "將文本轉換為語音中...")
    tts_start_time = time.time()
    try:
        timestamp = int(time.time())
        audio_file = f"./outputs/audio_{timestamp}.mp3"
        audio_path = text_to_speech(
            text=llm_output,
            output_file=audio_file,
            voice=voice_id
        )
        tts_time = time.time() - tts_start_time
        llm_output_with_time += f"\n[TTS 耗時: {format_time(tts_time)}]"
    except Exception as e:
        tts_time = time.time() - tts_start_time
        total_time = time.time() - total_start_time
        error_msg = f"{llm_output_with_time}\nTTS 錯誤: {str(e)}\n[耗時: {format_time(tts_time)}]"
        return error_msg, None, None, {"llm": llm_time, "tts": tts_time, "video": 0, "total": total_time}
    
    # 步驟 3: 生成視頻
    progress(0.6, "生成嘴型同步視頻中...")
    video_start_time = time.time()
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
            whisper_dir=MODEL_PATHS["whisper_dir"],
            use_preloaded_models=True  # 使用預先載入的模型
        )
        video_time = time.time() - video_start_time
        total_time = time.time() - total_start_time
        
        llm_output_with_time += f"\n[視頻生成耗時: {format_time(video_time)}]\n[總耗時: {format_time(total_time)}]"
    except Exception as e:
        video_time = time.time() - video_start_time
        total_time = time.time() - total_start_time
        error_msg = f"{llm_output_with_time}\n視頻生成錯誤: {str(e)}\n[耗時: {format_time(video_time)}]\n[總耗時: {format_time(total_time)}]"
        return error_msg, audio_path, None, {"llm": llm_time, "tts": tts_time, "video": video_time, "total": total_time}
    
    progress(1.0, "處理完成!")
    
    # 返回處理結果和各部分耗時
    time_stats = {
        "llm": llm_time,
        "tts": tts_time,
        "video": video_time,
        "total": total_time
    }
    
    return llm_output_with_time, audio_path, video_path, time_stats

def build_interface():
    """
    構建 Gradio 界面
    """
    with gr.Blocks(title="MuseTalk 數位司儀系統", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# MuseTalk 數位司儀系統")
        gr.Markdown("輸入您的問題，系統將生成文本回應並創建對應的視頻")
        
        if models_loaded:
            gr.Markdown(f"✅ MuseTalk 模型已預先載入，視頻生成速度將大幅提升 (載入耗時: {format_time(model_load_time)})")
        else:
            gr.Markdown("⚠️ MuseTalk 模型預載入失敗，將使用標準模式")
        
        # 第一行：輸入區域
        with gr.Row():
            query_input = gr.Textbox(
                label="您的問題", 
                placeholder="請輸入您想問的問題...",
                lines=3,
                value=""  # 清空初始值
            )
        
        # 第二行：選項和提交按鈕
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
        
        # 第三行：輸出區域 - AI文本和音頻在左側，視頻在右側
        with gr.Row():
            # 左側：文本和音頻
            with gr.Column(scale=1):
                # 輸出區域 - 預設顯示歡迎內容
                welcome_text_with_time = f"歡迎使用數位司儀系統\n\n[TTS 耗時: {format_time(welcome_tts_time)}]\n[視頻生成耗時: {format_time(welcome_video_time)}]"
                llm_output = gr.Textbox(
                    label="AI 回應文本", 
                    lines=5,
                    value=welcome_text_with_time
                )
                audio_output = gr.Audio(
                    label="生成的音頻",
                    value=welcome_audio if welcome_audio else None
                )
                
                # 添加耗時統計圖表
                time_stats = gr.Json(label="處理耗時統計", visible=False)
                
            # 右側：視頻
            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="生成的視頻",
                    value=welcome_video if welcome_video else None
                )
        
        # 處理提交
        submit_btn.click(
            fn=process_query,
            inputs=[
                query_input,
                template_dropdown,
                voice_dropdown
            ],
            outputs=[
                llm_output,
                audio_output, 
                video_output,
                time_stats
            ]
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
        
        💡 **優化提示**: 系統已預先載入模型，首次生成視頻後，後續處理速度會大幅提升！
        """)
        
    return interface

# 啟動 Gradio 界面
if __name__ == "__main__":
    interface = build_interface()
    interface.launch(share=True)