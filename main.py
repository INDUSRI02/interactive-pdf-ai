import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Prompts
# -----------------------------
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Generate answers from the data in the vector store itself only if it’s there in the document. Else, say "There is no information about it."
"""

instruction = "Answer the question you asked: \n\n {text}"
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST
prompt = PromptTemplate(template=template, input_variables=["text"])

# -----------------------------
# Functions
# -----------------------------
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_text(raw_text)

def make_vector_store(textChunks):
    try:
        embeddings = HuggingFaceHubEmbeddings()
        vectordb = Chroma.from_texts(textChunks, embedding=embeddings)
        return vectordb
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def make_conversational_chain(vector_store):
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, prompt=prompt)
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conv_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def user_input(user_question):
    if not st.session_state.conversation:
        st.warning("Please process the PDFs first!")
        return
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        st.write(f"{i+1}. {message['content']}")

# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
    st.header("Chat with Multiple PDFs 🐆")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Enter your question about the PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload Your PDF Documents")
        pdf_docs = st.file_uploader("Upload PDFs (multiple allowed)", accept_multiple_files=True, type=["pdf"])

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = make_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDFs.")
                        return

                    st.write("Raw text length:", len(raw_text))
                    st.write(raw_text[:500])  # preview

                    textChunks = text_chunks(raw_text)
                    st.write("Number of text chunks:", len(textChunks))
                    if len(textChunks) == 0:
                        st.error("Text could not be split into chunks.")
                        return

                    vector_store = make_vector_store(textChunks)
                    if vector_store:
                        st.session_state.conversation = make_conversational_chain(vector_store)
                        st.success("PDFs processed successfully! You can now ask questions.")
                    else:
                        st.error("Failed to create vector store.")

        st.markdown("---")
        st.caption("💡 Built with ❤️ by Indu Sri")

if __name__ == "__main__":
    main()
