import os
import json
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="The attention mask and the pad token id were not set"
)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory 


with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))) as config_file:
    config = json.load(config_file)
MODEL_DIR = os.path.join(config["base_dir"], config["model_dir"])


def load_model_and_tokenizer(model_dir, device):
    """
    Takes the available device and model_dir to load the LLM and Tokenizer

    Returns: model, tokenizer object from the AutoModel and AutoTokenizer subclasses.
    """
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

def initialize_conversation_dependencies():
    """
    This function creates a huggingface pipeline using the model and the tokenizer

    Returns: LLMChain object.
    """
    
    # Memory for the conversation
    memory = ConversationBufferMemory()

    # template and prompt for the conversation
    template = "History: {history}\n\nHuman: {input}\nAI: "
    prompt = PromptTemplate(template=template, input_variables=["history","input"])

    return prompt, memory


def chat(model, tokenizer):

    print("Initializing the conversation...")
    prompt, memory = initialize_conversation_dependencies()

    print("Start chatting! (Type 'exit' to end the conversation)")
    
    
    history = ''
    while True:
        user_input = input("Human: ")
        
        if user_input.lower() == "exit":
            break
        elif user_input.strip() == "":
            continue
        
        prompt_text = prompt.format(history=history, input=user_input)

        ai_response = generate_text(model, tokenizer, prompt_text)
        
        # Saving the conversation..
        memory.chat_memory.add_user_message(user_input)
        history += "Human: "+ user_input + "\n" + "AI: "+ ai_response + "\n"
        if ai_response != None:
            memory.chat_memory.add_ai_message(ai_response)
        print(f"AI: {ai_response}")

    print(f"\n\n\n\n\n\n##### Printing History again ####\n{history}")

def generate_text_stub(model, tokenizer, input_text):
    return "This is a stub response"

def generate_text(model, tokenizer, input_text)->str:
    """
    Uses the model and tokenizer to generate text based on some input_text.

    Returns: Text.
    """
    try:
        inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        attention_mask = torch.ones(inputs.shape, device=device)
        pad_token_id = tokenizer.eos_token_id

        output_ids = model.generate(
            inputs,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id, 
            #max_length=50, 
            max_new_tokens=50,
            do_sample=True, 
            top_p=0.95, 
            top_k=100, 
            num_return_sequences=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens = True)

        return output_text
    
    except RuntimeError as rue:
        print(f"RuntimeError : {rue}")
        return ''
    

if __name__ == "__main__":
    # load the model
    print(f"Loading the model from : {MODEL_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device selected : {device}")
    
    
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR, device)
    print("Model loaded successfully..")

    chat(model, tokenizer)