# main.py
# This script integrates the LLM, TTS, and video generation modules to create a complete pipeline.

import os
import argparse
import time
import datetime
from sources.llm_module import generate_text, get_model_and_tokenizer
from sources.tts_module import text_to_speech, get_taiwanese_voices
from sources.video_module import generate_video, preload_models

def format_time(seconds):
    """將秒數格式化為易讀的時間格式"""
    return str(datetime.timedelta(seconds=round(seconds)))

def preload_all_models(args):
    """
    預載入所有模型，包括 LLM、TTS 和 MuseTalk 模型，避免每次處理時重新載入
    
    Args:
        args: 命令行參數
        
    Returns:
        dict: 包含預載入模型的狀態和耗時信息
    """
    preload_stats = {
        "llm": {"loaded": False, "time": 0},
        "musetalk": {"loaded": False, "time": 0},
    }
    
    print("預載入所有模型中...")
    
    # 1. 預載入 LLM 模型
    print("1. 載入大語言模型 (LLM)...")
    llm_start_time = time.time()
    try:
        tokenizer, model = get_model_and_tokenizer()
        if tokenizer is not None and model is not None:
            preload_stats["llm"]["loaded"] = True
            print("   ✓ LLM 模型載入成功")
        else:
            print("   ✗ LLM 模型載入失敗")
    except Exception as e:
        print(f"   ✗ LLM 模型載入出錯: {str(e)}")
    
    preload_stats["llm"]["time"] = time.time() - llm_start_time
    print(f"   LLM 載入耗時: {format_time(preload_stats['llm']['time'])}")
    
    # 2. 預載入 MuseTalk 模型
    print("2. 載入 MuseTalk 模型...")
    musetalk_start_time = time.time()
    try:
        musetalk_loaded = preload_models(
            unet_model_path=args.unet_model_path,
            unet_config=args.unet_config,
            whisper_dir=args.whisper_dir,
            version="v15",
            extra_margin=args.extra_margin
        )
        preload_stats["musetalk"]["loaded"] = musetalk_loaded
        if musetalk_loaded:
            print("   ✓ MuseTalk 模型載入成功")
        else:
            print("   ✗ MuseTalk 模型載入失敗")
    except Exception as e:
        print(f"   ✗ MuseTalk 模型載入出錯: {str(e)}")
    
    preload_stats["musetalk"]["time"] = time.time() - musetalk_start_time
    print(f"   MuseTalk 載入耗時: {format_time(preload_stats['musetalk']['time'])}")
    
    # 3. TTS 模型不需要額外預載，EdgeTTS 會在運行時連接服務
    print("3. TTS 模型採用 EdgeTTS 在線服務，無需預載")
    
    # 計算總耗時
    total_time = preload_stats["llm"]["time"] + preload_stats["musetalk"]["time"]
    print(f"所有模型預載入完成，總耗時: {format_time(total_time)}")
    
    return preload_stats

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='MuseTalk: Text to Talking Video Pipeline')
    parser.add_argument('--input', type=str, help='Input text for LLM', default="你好，請介紹一下台灣的美食文化")
    parser.add_argument('--skip_llm', action='store_true', help='Skip LLM step')
    parser.add_argument('--skip_tts', action='store_true', help='Skip TTS step')
    parser.add_argument('--skip_preload', action='store_true', help='Skip model preloading')
    parser.add_argument('--audio_file', type=str, help='Path to audio file (if skipping LLM/TTS)', default="output_audio.mp3")
    parser.add_argument('--video_reference', type=str, help='Path to reference video/image', default="assets/demo/sun1/sun.png")
    parser.add_argument('--output_dir', type=str, help='Output directory', default="./outputs")
    parser.add_argument('--tts_voice', type=str, help='TTS voice to use', default="zh-TW-HsiaoChenNeural")
    
    # MuseTalk specific arguments
    parser.add_argument('--unet_model_path', type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument('--unet_config', type=str, default="./models/musetalkV15/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument('--whisper_dir', type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument('--fps', type=int, default=25, help="Video frames per second")
    parser.add_argument('--extra_margin', type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for video generation")
    parser.add_argument('--welcome_message', action='store_true', help="Generate welcome message on startup", default=True)
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 預載入所有模型
    preload_info = None
    if not args.skip_preload:
        preload_info = preload_all_models(args)
    
    # 生成歡迎訊息（可選）
    if args.welcome_message and preload_info and preload_info["musetalk"]["loaded"]:
        welcome_text = "歡迎使用數位司儀系統"
        welcome_audio_path = os.path.join(args.output_dir, "welcome_audio.mp3")
        welcome_video_path = os.path.join(args.output_dir, "welcome_video.mp4")
        
        print("\n生成歡迎訊息...")
        try:
            # 生成歡迎語音
            audio_file = text_to_speech(
                text=welcome_text,
                output_file=welcome_audio_path,
                voice=args.tts_voice
            )
            print(f"歡迎語音生成成功: {audio_file}")
            
            # 生成歡迎視頻
            video_file = generate_video(
                audio_file=audio_file,
                video_path=args.video_reference,
                output_path=welcome_video_path,
                use_preloaded_models=True  # 使用預先載入的模型
            )
            print(f"歡迎視頻生成成功: {video_file}")
        except Exception as e:
            print(f"生成歡迎訊息時出錯: {str(e)}")
    
    # Final paths
    output_text_path = os.path.join(args.output_dir, "generated_text.txt")
    output_audio_path = os.path.join(args.output_dir, "output_audio.mp3")
    output_video_path = os.path.join(args.output_dir, "output_video.mp4")
    
    try:
        # Step 1: Generate text using LLM (if not skipped)
        if not args.skip_llm:
            print("\nStep 1: Generating text using LLM...")
            start_time = time.time()
            generated_text = generate_text(args.input)
            llm_time = time.time() - start_time
            print(f"Generated Text: {generated_text} (耗時: {format_time(llm_time)})")
            
            # Save generated text
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
        else:
            # If LLM step is skipped, use the input text directly
            generated_text = args.input
            print(f"Using provided text: {generated_text}")
            
        # Step 2: Convert text to speech using TTS (if not skipped)
        if not args.skip_tts:
            print(f"\nStep 2: Converting text to speech using TTS with voice {args.tts_voice}...")
            start_time = time.time()
            audio_file = text_to_speech(
                text=generated_text, 
                output_file=output_audio_path,
                voice=args.tts_voice
            )
            tts_time = time.time() - start_time
            print(f"Audio file generated: {audio_file} (耗時: {format_time(tts_time)})")
        else:
            # If TTS step is skipped, use the provided audio file
            audio_file = args.audio_file
            print(f"Using provided audio file: {audio_file}")

        # Step 3: Generate video synchronized with audio
        print("\nStep 3: Generating video synchronized with audio...")
        start_time = time.time()
        
        # 檢查是否可以使用預載入的模型
        use_preloaded = preload_info and preload_info["musetalk"]["loaded"] and not args.skip_preload
        
        video_file = generate_video(
            audio_file=audio_file,
            video_path=args.video_reference,
            bbox_shift=0,
            extra_margin=args.extra_margin,
            output_path=output_video_path,
            fps=args.fps,
            batch_size=args.batch_size,
            unet_model_path=args.unet_model_path,
            unet_config=args.unet_config,
            whisper_dir=args.whisper_dir,
            use_preloaded_models=use_preloaded  # 使用預先載入的模型
        )
        video_time = time.time() - start_time
        print(f"Video file generated: {video_file} (耗時: {format_time(video_time)})")
        
        # 顯示總處理時間
        print("\n處理完成!")
        if not args.skip_llm and not args.skip_tts:
            total_time = llm_time + tts_time + video_time
            print(f"總處理時間: {format_time(total_time)}")
            print(f"- LLM 處理時間: {format_time(llm_time)}")
            print(f"- TTS 處理時間: {format_time(tts_time)}")
            print(f"- 視頻生成時間: {format_time(video_time)}")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()