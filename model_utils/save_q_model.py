import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific warnings by category
# Example: warnings.filterwarnings("ignore", category=UserWarning)


with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))) as config_file:
    config = json.load(config_file)
MODEL_DIR = os.path.join(config["base_dir"], config["model_dir"])
Q_MODEL_DIR = os.path.join(config["base_dir"], config["q_model_dir"])
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_quantized_model(model_dir, save_dir):
    """
    Save the quantized model for faster loading for subsequent uses.  
    """
    assert model_dir!=None and save_dir!=None
    qnt_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading the model on cpu and saving at {save_dir}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            quantization_config=qnt_config)
        
        model.save_pretrained(save_dir)

        print(f"Quantized Model saved to {save_dir}")
    except RuntimeError as rue:
        print(f"RuntimeError : {rue}")
        
def test_model():
    """
    Loading the model Quantized model saved in the directory specified in the "q_model_dir" field of the config file 
    """
    assert Q_MODEL_DIR!=None
    
    try:
        # Loading the model
        tokenizer = AutoTokenizer.from_pretrained(Q_MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token' : '<|PAD|>'})
            
        model = AutoModelForCausalLM.from_pretrained(
            Q_MODEL_DIR, 
            device_map = device
            )
        #model.config.pad_token_id = tokenizer.pad_token_id
        # Running Inference on a test query
        query = "Hello"
        input_ids = tokenizer.encode(query, return_tensors="pt").to(model.device) # type: ignore
        #attention_mask = torch.ones(input_ids.shape).to(model.device)
        pad_token_id = tokenizer.pad_token_id

        output_ids = model.generate(
            input_ids,
            #attention_mask = attention_mask,
            pad_token_id = pad_token_id,
            max_new_tokens = 10,
            num_return_sequences = 1
        )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except RuntimeError as rerror:
        print(f"There was an error in loading and inferencing the model: \n{rerror}")
    except UserWarning as uwarning:
        pass

    print(f"The model ran the test successfully and is ready to be used!.\nTest Prompt: {query}\nTest Output text: {output}")



def parse_arguments():
    """
    Parses the command line arguments

    Returns: 
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="The location of the original model to be quantized")
    parser.add_argument(
        "--model_dir",
        type = str,
        required = False,
        help = "The name of the model to be downloaded from Huggingface"
    )

    parser.add_argument(
        "--save_directory",
        type = str,
        required = False,
        help = "The (relative)location where the model is to be saved"
    )

    parser.add_argument(
        "--test",
        type = bool,
        required = False,
        help = "Testing the model is saved correctly or not. Note: The model dir will be fetched from the config file."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.test == False:
        save_quantized_model(args.model_dir, args.save_directory)

    else:
        print("Testing saved model with a simple query")
        test_model()