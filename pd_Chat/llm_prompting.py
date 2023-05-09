import os
from pathlib import Path
from functools import partial
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.llms import OpenAI
import openai

def create_bot(API_O : str, MODEL : str, K : int, ENTITY_MEMORY_CONVERSATION_TEMPLATE):
    # Create an OpenAI instance
    llm = OpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 

    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)
        
        # Create the ConversationChain object with the specified configuration
        Conversation = ConversationChain(
                llm=llm, 
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                memory=st.session_state.entity_memory
            )  
        return Conversation
    else:
        pass

def retrieve_ai_answer(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


retrieve_better_title = lambda text : "[Limit: < 50 tokens] Can you provide a more engaging title for this paper? Think about a title that would be engaging and provocative." + f"\n\n" + text

retrieve_paper_text = lambda text : "[Limit: 500 tokens] Can you provide the abstract, introduction, methods, results and conclusions in 5-10 bullet points?." + f"\n\n" + text

retrieve_significance = lambda text : "[Limit: 75 tokens] Can you provide a persuasive significance statement. 3 reasons why looking at this topic is important?" + f"\n\n" + text

retrieve_key_references = lambda text : "[Limit: 100 tokens] Can you provide a list of the 5 key references for this paper listed in the references? Please include the title, authors, journal, year and DOI, and possibly the link if it is present. I do not need any external links just use the information in the references. Limit to 100 words. Abridge the citations if necessary" + f"\n\n" + text

