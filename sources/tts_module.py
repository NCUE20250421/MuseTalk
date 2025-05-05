# tts_module.py
# This module handles text-to-speech conversion using EdgeTTS for better Taiwanese accent.

import asyncio
import edge_tts
import os
import re

def preprocess_text(text):
    """
    預處理文本，移除 "Assistant:" 和 "Human:" 標記及其後面的內容。
    還會清理多餘的空白行和格式化問題。

    Args:
        text (str): 輸入的文本

    Returns:
        str: 處理後的文本
    """
    # 移除 "Assistant:" 和 "Human:" 標記
    text = re.sub(r'^\s*Assistant:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Human:\s*', '', text, flags=re.MULTILINE)
    
    # 移除可能存在的 Markdown 代碼塊標記
    text = re.sub(r'```[\w]*\n|```\n?', '', text)
    
    # 合併連續的空行為一個空行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # 去除開頭和結尾的空白
    text = text.strip()
    
    return text

async def _text_to_speech_async(text, output_file="output_audio.mp3", voice="zh-TW-HsiaoChenNeural", rate="+0%", volume="+0%"):
    """
    Asynchronous function to convert text to speech using EdgeTTS.
    """
    # 預處理文本，移除 Assistant: 和 Human: 標記
    cleaned_text = preprocess_text(text)
    
    communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate, volume=volume)
    await communicate.save(output_file)

def text_to_speech(text, output_file="output_audio.mp3", voice="zh-TW-HsiaoChenNeural", rate="+0%", volume="+0%"):
    """
    Converts text to speech using EdgeTTS with Taiwanese accent.

    Args:
        text (str): The text to convert to speech.
        output_file (str): The path to save the generated audio file.
        voice (str): The voice to use. Options for Taiwanese accent:
                     - zh-TW-HsiaoChenNeural (female)
                     - zh-TW-YunJheNeural (male)
                     - zh-TW-HsiaoYuNeural (female child)
        rate (str): Speaking rate adjustment. e.g., "+10%", "-10%"
        volume (str): Volume adjustment. e.g., "+10%", "-10%"

    Returns:
        str: The path to the generated audio file.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_text_to_speech_async(text, output_file, voice, rate, volume))
    finally:
        loop.close()
    
    print(f"Audio file saved to {output_file}")
    return output_file

async def list_voices():
    """
    Lists available voices in EdgeTTS.
    """
    voices = await edge_tts.list_voices()
    taiwanese_voices = [v for v in voices if v["Locale"].startswith("zh-TW")]
    return taiwanese_voices

def get_taiwanese_voices():
    """
    Returns a list of available Taiwanese voices.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        voices = loop.run_until_complete(list_voices())
    finally:
        loop.close()
    
    return voices