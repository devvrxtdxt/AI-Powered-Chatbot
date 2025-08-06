import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load your Groq API key securely
with open("groq_api.txt", "r") as f:
    groq_api_key = f.read().strip()

st.title("Chat with Your PDF (Groq RAG Chatbot)")

uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

if uploaded_file:
    # Save temp file for processing
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # 1. Load and split PDF into chunks
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    
    # 2. Embedding and retrieval
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # 3. Connect to LLM (Groq Llama-3)
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # 4. User Q&A
    question = st.text_input("Ask a question about your document:")
    if question:
        answer = qa_chain.run(question)
        st.write(answer)
