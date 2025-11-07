import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import tempfile

# Load your Groq API key securely - works for both local and cloud deployment
def get_groq_api_key():
    # First try environment variable (for cloud deployment)
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        st.success("✅ Using API key from environment variable")
        return api_key
    
    # Fallback to local file (for local development)
    try:
        # Try different possible file paths
        possible_paths = ["groq_api.txt", "./groq_api.txt", os.path.join(os.getcwd(), "groq_api.txt")]
        
        for file_path in possible_paths:
            try:
                with open(file_path, "r") as f:
                    api_key = f.read().strip()
                    if api_key:
                        st.success(f"✅ Using API key from {file_path}")
                        return api_key
            except FileNotFoundError:
                continue
        
        # If we get here, no file was found
        st.error(f"❌ Groq API key not found!")
        st.error("Please either:")
        st.error("1. Set GROQ_API_KEY environment variable, or")
        st.error("2. Create a groq_api.txt file in the project root")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Files in current directory: {os.listdir('.')}")
        st.stop()
        
    except Exception as e:
        st.error(f"❌ Error reading API key: {str(e)}")
        st.stop()

groq_api_key = get_groq_api_key()

st.title("Chat with Your PDF (Groq RAG Chatbot)")

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

# Process PDF if uploaded and not already processed
if uploaded_file:
    # Check if we need to process (new file or first time)
    if st.session_state.vectorstore is None:
        try:
            # Save temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_pdf_path = tmp_file.name
            
            # 1. Load and split PDF into chunks
            with st.spinner("Loading and processing PDF..."):
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = splitter.split_documents(documents)
            
            # 2. Embedding and retrieval using FAISS (more cloud-friendly)
            with st.spinner("Creating embeddings..."):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Use FAISS instead of ChromaDB for better cloud compatibility
                vectorstore = FAISS.from_documents(docs, embeddings)
            
            # 3. Setup retriever and LLM
            with st.spinner("Setting up AI model..."):
                # Initialize the LLM
                llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0)
                
                # Create retriever for document search
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # Store in session state so we can reuse across questions
                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.session_state.vectorstore = vectorstore
            
            st.success("✅ Document processed successfully! You can now ask questions.")
            
            # Clean up temporary files
            try:
                os.unlink(temp_pdf_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.error("Please try uploading a different PDF file.")
    else:
        st.info("✅ Document already processed. You can ask questions below.")

# User Q&A section
if st.session_state.vectorstore is not None and st.session_state.llm is not None:
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Retrieve relevant documents using vectorstore similarity search (most reliable method)
                relevant_docs = st.session_state.vectorstore.similarity_search(question, k=4)
                
                # Combine document contents into context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Create prompt with context
                prompt = f"""Use the following context from the document to answer the question.
If you don't know the answer based on the context, say "I don't know" or "The context doesn't provide enough information."
Keep your answer concise and based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""
                
                # Get answer from LLM
                response = st.session_state.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                st.write("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
elif uploaded_file is None:
    st.info("👆 Please upload a PDF document to get started.")