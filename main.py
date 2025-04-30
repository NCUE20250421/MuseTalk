# main.py
# This script integrates the LLM, TTS, and video generation modules to create a complete pipeline.

from llm_module import generate_text
from tts_module import text_to_speech
from video_module import generate_video

def main():
    # Step 1: Generate text using LLM
    # input_text = "你好，今天的天氣如何？"
    # generated_text = generate_text(input_text)
    # print(f"Generated Text: {generated_text}")

    # Step 2: Convert text to speech using TTS
    # audio_file = text_to_speech(generated_text)
    # print(f"Audio file generated: {audio_file}")

    # Step 3: Generate video synchronized with audio
    video_file = generate_video("data\audio\10sec.wav")
    print(f"Video file generated: {video_file}")

if __name__ == "__main__":
    main()