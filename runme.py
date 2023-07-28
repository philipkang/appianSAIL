import pprint
import streamlit as st
import tempfile
import os
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.document_loaders import JSONLoader
import json
from pathlib import Path 
from pprint import pprint

DRIVE_FOLDER="./data"
loader = DirectoryLoader(DRIVE_FOLDER, glob='**/*.json', loader_cls=TextLoader)

# loader = JSONLoader(file_path="data/doc_text.json", jq_schema='.content', json_lines=True) # Use this line if you only need data.json
data = loader.load()
# pprint(data)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
split_docs = text_splitter.split_documents(data)

# vectorstore = FAISS.from_documents(data, embeddings)
vectorstore = FAISS.from_documents(split_docs, embeddings)

chain = ConversationalRetrievalChain.from_llm(
llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo'),
retriever=vectorstore.as_retriever())

def conversational_chat(query):
        
        result = chain({"question": query, 
        "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
 #   st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]
    st.session_state['generated'] = ["Hello ! Ask me anything about the vehicle maintenance requests "]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
        
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
            
        user_input = st.text_input("Query:", placeholder="Talk about your data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
            
    if submit_button and user_input:
        output = conversational_chat(user_input)
            
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)


    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

