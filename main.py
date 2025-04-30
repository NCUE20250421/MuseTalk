# main.py
# This script integrates the LLM, TTS, and video generation modules to create a complete pipeline.

import os
import argparse
from llm_module import generate_text
from tts_module import text_to_speech, get_taiwanese_voices
from video_module import generate_video

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='MuseTalk: Text to Talking Video Pipeline')
    parser.add_argument('--input', type=str, help='Input text for LLM', default="你好，請介紹一下台灣的美食文化")
    parser.add_argument('--skip_llm', action='store_true', help='Skip LLM step')
    parser.add_argument('--skip_tts', action='store_true', help='Skip TTS step')
    parser.add_argument('--audio_file', type=str, help='Path to audio file (if skipping LLM/TTS)', default="output_audio.mp3")
    parser.add_argument('--video_reference', type=str, help='Path to reference video/image', default="data/video/sun.mp4")
    parser.add_argument('--output_dir', type=str, help='Output directory', default="./outputs")
    parser.add_argument('--tts_voice', type=str, help='TTS voice to use', default="zh-TW-HsiaoChenNeural")
    
    # MuseTalk specific arguments
    parser.add_argument('--unet_model_path', type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument('--unet_config', type=str, default="./models/musetalkV15/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument('--whisper_dir', type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument('--fps', type=int, default=25, help="Video frames per second")
    parser.add_argument('--extra_margin', type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for video generation")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Final paths
    output_text_path = os.path.join(args.output_dir, "generated_text.txt")
    output_audio_path = os.path.join(args.output_dir, "output_audio.mp3")
    output_video_path = os.path.join(args.output_dir, "output_video.mp4")
    
    try:
        # Step 1: Generate text using LLM (if not skipped)
        if not args.skip_llm:
            print("Step 1: Generating text using LLM...")
            generated_text = generate_text(args.input)
            print(f"Generated Text: {generated_text}")
            
            # Save generated text
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
        else:
            # If LLM step is skipped, use the input text directly
            generated_text = args.input
            print(f"Using provided text: {generated_text}")
            
        # Step 2: Convert text to speech using TTS (if not skipped)
        if not args.skip_tts:
            print(f"Step 2: Converting text to speech using TTS with voice {args.tts_voice}...")
            audio_file = text_to_speech(
                text=generated_text, 
                output_file=output_audio_path,
                voice=args.tts_voice
            )
            print(f"Audio file generated: {audio_file}")
        else:
            # If TTS step is skipped, use the provided audio file
            audio_file = args.audio_file
            print(f"Using provided audio file: {audio_file}")

        # Step 3: Generate video synchronized with audio
        print("Step 3: Generating video synchronized with audio...")
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
            whisper_dir=args.whisper_dir
        )
        print(f"Video file generated: {video_file}")
            
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()