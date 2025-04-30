# llm_module.py
# This module handles text generation using a large language model (LLM).

import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMConfig:
    """Configuration for the LLM module"""
    MODEL_NAME = "Qwen/Qwen-7B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 512  # 減少最大長度以產生更簡短的回答
    TEMPERATURE = 0.7
    TOP_P = 0.9
    # 修改提示模板，讓模型知道我們需要簡短對話式的回答
    DEFAULT_PROMPT_TEMPLATE = """Human: {input_text}
Assistant: 請用簡短的一兩句話回答，像對話一樣。"""

# Initialize tokenizer and model
def initialize_model(model_name=None):
    """
    Initialize the LLM model and tokenizer.

    Args:
        model_name (str, optional): Name or path of the model to load. Defaults to LLMConfig.MODEL_NAME.

    Returns:
        tuple: (tokenizer, model) or (None, None) if initialization fails
    """
    try:
        print(f"Loading model: {model_name or LLMConfig.MODEL_NAME}")
        model_name = model_name or LLMConfig.MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            device_map=LLMConfig.DEVICE
        )
        print(f"Model loaded successfully on {LLMConfig.DEVICE}")
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None, None

# Lazy loading of model and tokenizer
_tokenizer = None
_model = None

def get_model_and_tokenizer():
    """Get or initialize model and tokenizer"""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer, _model = initialize_model()
    return _tokenizer, _model

def extract_first_sentence(text):
    """
    從文本中提取第一個完整的句子。
    
    Args:
        text (str): 輸入文本。
        
    Returns:
        str: 提取的第一個句子，如果沒有找到句子則返回原文本。
    """
    # 定義中文和英文的句子結束標記
    sentence_endings = r'[。！？!?;；]'
    
    # 查找第一個句子
    match = re.search(f'(.+?{sentence_endings})', text)
    if match:
        return match.group(1).strip()
    
    # 如果沒找到句號等結束標記，則返回第一行文本
    lines = text.split('\n')
    if lines and lines[0].strip():
        return lines[0].strip()
    
    # 最後退回到返回原始文本
    return text.strip()

def generate_text(input_text, prompt_template=None, max_length=None, temperature=None, top_p=None):
    """
    Generates text using the Qwen model and extracts only the Assistant's response.

    Args:
        input_text (str): The input text provided by the user.
        prompt_template (str, optional): Custom prompt template to use.
        max_length (int, optional): Maximum length of generated text.
        temperature (float, optional): Temperature for sampling.
        top_p (float, optional): Top-p sampling parameter.

    Returns:
        str: The Assistant's response extracted from the generated text, or a default response if extraction fails.
    """
    tokenizer, model = get_model_and_tokenizer()
    
    if tokenizer is None or model is None:
        return "模型初始化失敗，請檢查錯誤日誌。"
    
    try:
        # Use default values if not provided
        prompt_template = prompt_template or LLMConfig.DEFAULT_PROMPT_TEMPLATE
        max_length = max_length or LLMConfig.MAX_LENGTH
        temperature = temperature or LLMConfig.TEMPERATURE
        top_p = top_p or LLMConfig.TOP_P
        
        # Format the prompt with the input text
        prompt = prompt_template.format(input_text=input_text)
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(LLMConfig.DEVICE)
        
        # Generate text
        outputs = model.generate(
            **inputs, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the Assistant's response
        if "Assistant:" in generated_text:
            assistant_response = generated_text.split("Assistant:", 1)[1].strip()
        else:
            # If we can't find the Assistant marker, return the text after the input
            # This is a fallback for models with different formats
            if input_text in generated_text:
                assistant_response = generated_text.split(input_text, 1)[1].strip()
            else:
                # Final fallback: return everything
                assistant_response = generated_text
        
        # 清除提示中的指令
        assistant_response = assistant_response.replace("請用簡短的一兩句話回答，像對話一樣。", "")
        
        # 提取第一個完整句子作為回答
        first_sentence = extract_first_sentence(assistant_response)
        
        return first_sentence
    
    except Exception as e:
        print(f"Error generating text: {str(e)}")
        return "我無法處理您的請求，發生了錯誤。"