import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

from ctransformers import AutoModelForCausalLM

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Prompts
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe.
Generate answers from the data in the vector store itself only if it’s there in the document. Else, say "There is no information about it."
"""

instruction = "Answer the question you asked: \n\n {text}"
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST
prompt = PromptTemplate(template=template, input_variables=["text"])

# --- Functions ---

def make_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return splitter.split_text(raw_text)

def make_vector_store(textChunks):
    try:
        embeddings = HuggingFaceHubEmbeddings()
        return FAISS.from_texts(textChunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def make_conversational_chain(vector_store):
    try:
        llm = CTransformers(
            model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
            model_type='llama',
            config={'max_new_tokens': 4096, 'context_length': 4096, 'temperature': 0.01}
        )
    except Exception as e:
        st.error("Error loading LLaMA model. Make sure model exists.")
        return None

    memory = ConversationBufferMemory(memory_key="chat_history", prompt=prompt, return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)

def user_input(user_question):
    if not st.session_state.get("conversation"):
        st.warning("Please process PDFs first!")
        return
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chatHistory = response["chat_history"]
    for i, msg in enumerate(st.session_state.chatHistory):
        st.write(f"{msg['role'].capitalize()}: {msg['content']}")

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs 🐆")

    user_question = st.text_input("Enter your question about the PDF(s):")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chatHistory = None

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your PDF documents")
        pdf_docs = st.file_uploader("Upload PDFs (2 or more)", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if len(pdf_docs) < 2:
                st.warning("Please upload at least 2 PDFs!")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = make_pdf_text(pdf_docs)
                    chunks = text_chunks(raw_text)
                    vector_store = make_vector_store(chunks)
                    if vector_store:
                        st.session_state.conversation = make_conversational_chain(vector_store)
                        st.success("PDFs processed successfully!")
                    else:
                        st.error("Failed to create vector store.")

        st.markdown("---")
        st.caption("💡 Built with ❤️ by You")

if __name__ == "__main__":
    main()
