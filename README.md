# Conversational AI with Langchain

This project leverages Langchain and a pre-trained language model run locally to create a conversational AI chatbot. The chatbot is designed to engage in dynamic and context-aware interactions, using prompt templates and memory to maintain conversation history.

**Introduction**
The project aims to build a conversational AI chatbot using Langchain. The chabot is designed to handle multi-turns dialogues, maintain context, and provide coherent responses. 

**Methodology**
The methodology for developing this conversational AI chatbot involves several key components:
1. Model loading: 4-bit Quantizaztion is used to load the model on local GPU(low capacity).
2. Prompt Engineering: LangChain's 'PromptTenplate' is used to create structured prompts that guide the model's responses.
3. Context preservation: Langchain's 'ConversationBuffer' is employed to maintain the buffer of conversation history. This helps the model to generate context aware responses.
4. Conversation History Saving: At the end of each session, the conversation history is saved to a file. This allows for easy tracking and analysis of interactions.

**Upcoming**: 
1. Instruction tuning model to improve coherence in responses.
2. Processing the interactions in real-time to create long-term memories which can be later used to reference during future interactions.
