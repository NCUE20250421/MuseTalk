# tts_module.py
# This module handles text-to-speech conversion using Google TTS.

from gtts import gTTS
import os

def text_to_speech(text, output_file="output_audio.mp3"):
    """
    Converts text to speech using Google TTS.

    Args:
        text (str): The text to convert to speech.
        output_file (str): The path to save the generated audio file.

    Returns:
        str: The path to the generated audio file.
    """
    tts = gTTS(text)
    tts.save(output_file)
    print(f"Audio file saved to {output_file}")
    return output_file