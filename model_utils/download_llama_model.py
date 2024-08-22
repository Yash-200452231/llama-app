import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

def download_llama_model(model_name: str, save_directory: str):
    """
    Downloads the LLaMA 3 model and tokenizer from Hugging Face and saves them locally.

    Args:
        model_name (str): The name of the LLaMA 3 model on Hugging Face, e.g., "facebook/llama3".
        save_directory (str): The directory where the model and tokenizer will be saved.
    """
    # Ensure the save directory
    assert model_name != None and save_directory != None
    os.makedirs(save_directory, exist_ok=True)

    # Download model
    snapshot_download(repo_id = model_name, local_dir = save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

def parse_arguments():
    """
    Parses the command line arguments

    Returns: 
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Download the specified llama model from Huggingface")
    parser.add_argument(
        "--model_name",
        type = str,
        required = True,
        help = "The name of the model to be downloaded from Huggingface"
    )

    parser.add_argument(
        "--save_directory",
        type = str,
        required = True,
        help = "The (relative)location where the model is to be saved"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    download_llama_model(args.model_name, args.save_directory)