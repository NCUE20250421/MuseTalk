# llm_module.py
# This module handles text generation using a large language model (LLM).

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen model and tokenizer
model_name = "Qwen/Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Modify the generate_text function to return only the Assistant's response

def generate_text(input_text):
    """
    Generates text using the Qwen model and extracts only the Assistant's response.

    Args:
        input_text (str): The input text provided by the user.

    Returns:
        str: The Assistant's response extracted from the generated text, or a default response if extraction fails.
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the Assistant's response
    if "Assistant:" in generated_text:
        assistant_response = generated_text.split("Assistant:", 1)[1].strip()
        return assistant_response
    else:
        # Return a default meaningful response
        return "I'm sorry, I couldn't understand your request. Could you please rephrase it?"