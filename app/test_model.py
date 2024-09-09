import os
import json
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory 

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Loading all the necessary paths for various assets in the project directory
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))) as config_file:
    config = json.load(config_file)
MODEL_DIR = os.path.join(config["base_dir"], config["q_model_dir"])
CONVERSATION_HISTORY_DIR = os.path.join(config["base_dir"], config["conversation_history_dir"])


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

def initialize_conversation_dependencies():
    """
    This function creates a huggingface pipeline using the model and the tokenizer

    Returns: LLMChain object.
    """
    
    # Memory for the conversation
    memory = ConversationBufferMemory(memory_key='history')

    # template and prompt for the conversation
    template = "You are a chatbot. Your job is to chat with the user based on the current input which will be marked by 'Human:'. You should use the 'History:' as context to best answer the query in the current prompt.\nKeep the response simple and concise, and only respond as per the turn.\n\nHistory: {history}\n\nHuman: {input}\nAI: "
    prompt = PromptTemplate(template=template, input_variables=["history","input"])

    return prompt, memory

def save_conversation_history(history):
    #print(f"<{history}>")
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(CONVERSATION_HISTORY_DIR, filename)
    with open(filepath, "w") as file:
        file.write(history)

def chat(model, tokenizer):

    print("Initializing the conversation...")
    prompt, memory = initialize_conversation_dependencies()

    print("Start chatting! (Type 'exit' to end the conversation)")
    
    while True:
        history = memory.load_memory_variables({})["history"]
        #print(f"\n\n History : {history}")
        user_input = input("Human: ")
        
        
        if user_input.lower() == "exit":
            break
        elif user_input.strip() == "":
            continue
        
        
        ai_response = generate_text(model, tokenizer, prompt, history, user_input)
        #ai_response = generate_text_stub(model, tokenizer, prompt, history, user_input)
        print(f"AI: {ai_response}")

        # Add messages to memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(ai_response)
        
    #print(f"\n\n\n\n\n\n##### Printing History again ####\n{history}")
    save_conversation_history(history)
    print("Conversation history saved!")

def generate_text_stub(model, tokenizer, prompt, history, user_input):
    return "This is a stub response"

def generate_text(model, tokenizer, prompt, history, user_input)->str:
    """
    Uses the model and tokenizer to generate text based on some input_text.

    Returns: Text.
    """
    input_text = prompt.format(history= history, input= user_input)
    try:
        inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
        #attention_mask = torch.ones(inputs.shape, device=device)
        pad_token_id = tokenizer.pad_token_id

        output_ids = model.generate(
            inputs,
            #attention_mask=attention_mask,
            pad_token_id=pad_token_id, 
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