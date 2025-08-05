import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import LLMChain
from langchain.llms import CTransformers
from langchain.chains import  ConversationalRetrievalChain
from langchain.embeddings import  HuggingFaceHubEmbeddings
from transformers import AutoTokenizer
import transformers
import torch
from langchain import PromptTemplate
import warnings
warnings.filterwarnings("ignore")
from ctransformers import  AutoModelForCausalLM,AutoConfig
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import Docx2txtLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
generate answers  from the data in the vectore store  itself only if its there in the document else  tell "There is no information about it "."""

instruction = "Answer of the  Question  You asked  : \n\n {text}"

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

template = B_INST + SYSTEM_PROMPT + instruction + E_INST
prompt = PromptTemplate(template=template, input_variables=["text"])
#data in [pdfs is grouped
def make_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()


    return text


#splitting data into chunks
def text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    chunks = text_splitter.split_text(raw_text)
    return chunks

#embedding for text chunks and making knowledge base
def make_vector_store(textChunks):
    embeddings = HuggingFaceHubEmbeddings()
    vector_store = FAISS.from_texts(textChunks,embedding=embeddings)
    return vector_store

#conversational chain
#so here we will pass user question create embeddings for that question ,and we will try to find answer for that question in our knowledge base
#and rank those answers

def make_conversational_chain(vector_store):

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens':4096,
                                'context_length' : 4096,
                                'temperature': 0.01}
                        )
    memory = ConversationBufferMemory(memory_key ="chat_history",prompt = prompt, return_messages = True)
    conversational_chain = ConversationalRetrievalChain.from_llm(llm= llm, retriever=vector_store.as_retriever(),memory=memory)
    return conversational_chain


def user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.session_state.chatHistory = response["chat_history"]
    for i, message in enumerate(st.session_state.chatHistory):
        st.write(i,message)

#STREAMLIT CODE

def main():
    st.set_page_config("Chat with multiple PDFS")
    st.header("Chat with Multiple Pdfs 🐆")
    user_question =st.text_input("Enter You Questions regarding to the Pdf")
    if "conversation" not in st.session_state:
        st.session_state.conversation =None
    if "chat_history" not in st.session_state:
        st.session_state.chatHistory =None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings ")
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files and click on Process" , accept_multiple_files=True)
        if st.button("Process"):
           with st.spinner("processing"):
                raw_text = make_pdf_text(pdf_docs)
                textChunks = text_chunks(raw_text)
                vector_store = make_vector_store(textChunks)
                st.session_state.conversation = make_conversational_chain(vector_store)

                st.success("Done")
        st.title("credits to thala7️⃣")


if __name__ == "__main__":
    main()