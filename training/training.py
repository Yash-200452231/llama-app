import os
import json

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Loading all the necessary paths for various assets in the project directory
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))) as config_file:
    config = json.load(config_file)
MODEL_DIR = os.path.join(config["base_dir"], config["q_model_dir"])


def load_model_and_tokenizer(model_dir, device):
    """
    Takes the available device and model_dir to load the LLM and Tokenizer

    Returns: model, tokenizer object from the AutoModel and AutoTokenizer subclasses.
    """
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token' : '<|PAD|>'})
    
    # Loading quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map = device
    )

    return model, tokenizer


def add_adapters_to_model(model, lora_config=None):
    if lora_config==None:
        lora_config = LoraConfig(
               r = 4,
               target_modules=["q_proj", "o_proj"],
               bias="none",
               task_type="CASUAL_LM"
        )

    model.add_adapter(lora_config)
     

if __name__ == "__main__":
    print(f"Training lora adapters for base model at : {MODEL_DIR}")
    print(F"Searching for CUDA...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device selected for the model : {device}")

    print(f"Loading the model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, device)

    print(f"Attaching adapters to the model..")
    model = add_adapters_to_model(model)
    print(f"Adapters initialized successfully")

    print(f"Finding the training dataset...")

