import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import warnings

warnings.filterwarnings("ignore")

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Prompt template
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful, and honest assistant. 
Answer only based on the documents provided. 
If the answer is not in the documents, say "There is no information about it."
"""

instruction = "Answer the Question: \n\n{text}"
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(raw_text)

def make_vector_store(textChunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(textChunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def make_conversational_chain(vector_store):
    try:
        llm = CTransformers(
            model='models/llama-2-7b-chat.ggmlv3.q8_0.bin', 
            model_type='llama',
            config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 2048}
        )
    except Exception as e:
        st.error("Error loading model. Check model path and RAM availability.")
        return None

    memory = ConversationBufferMemory(memory_key="chat_history", prompt=prompt, return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)

def user_input(user_question):
    if not st.session_state.conversation:
        st.warning("Please process the PDFs first!")
        return
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chatHistory = response["chat_history"]
    for i, msg in enumerate(st.session_state.chatHistory):
        st.write(f"{i+1}. {msg['content']}")  # show messages cleanly

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs 🐆")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chatHistory = []

    user_question = st.text_input("Enter your question about the PDFs:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF files and click Process", accept_multiple_files=True)
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF!")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = make_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from PDFs.")
                        return
                    chunks = text_chunks(raw_text)
                    vector_store = make_vector_store(chunks)
                    if vector_store:
                        st.session_state.conversation = make_conversational_chain(vector_store)
                        st.success("PDFs processed successfully! Ask your questions above.")
                    else:
                        st.error("Failed to create vector store.")

        st.markdown("---")
        st.caption("💡 Built with ❤️ by Thala7️⃣")

if __name__ == "__main__":
    main()
