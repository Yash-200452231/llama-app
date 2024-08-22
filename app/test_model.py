import os
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))) as config_file:
    config = json.load(config_file)
MODEL_DIR = os.path.join(config["base_dir"], config["model_dir"])

def load_model_and_tokenizer(model_dir, device):
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Quantization config
    qnt_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Loading quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config = qnt_config,
        device_map = device
    )

    return model, tokenizer

def generate_text(model, tokenizer, input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    output_ids = model.generate(inputs, max_length = 50, do_sample = True, top_p = 0.95, top_k = 50, num_return_sequences = 2)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens = True)

    return output_text

if __name__ == "__main__":
    # load the model
    print(f"Loading the model from : {MODEL_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device selected : {device}")
    
    
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, device)
    print("Model loaded successfully..")

    print("Querying the model")
    input_query = "Write 2 lines about LLama3 model: "

    #Generation
    output_text = generate_text(model, tokenizer, input_query)
    print("Generated Text: ")
    print(output_text)