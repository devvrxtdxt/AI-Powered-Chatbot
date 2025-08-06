# RAG Groq Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, Groq, and Streamlit.  
**Upload a PDF and chat with your document!**

## Features

- 📄 Upload PDF documents
- 🤖 Chat with your documents using Groq's LLM
- 🔍 Semantic search and retrieval
- 💬 Interactive Streamlit interface

## Setup

### Prerequisites

- Python 3.8 or higher
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Groq API key:
   - Get your API key from [Groq Console](https://console.groq.com/)
   - Create a file named `groq_api.txt` in the project root
   - Add your API key to the file (just the key, no quotes)

5. Run the application:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Upload a PDF document using the file uploader
2. Wait for the document to be processed
3. Ask questions about your document in the text input
4. Get AI-powered answers based on your document content

## Deployment

### Local Development
```bash
streamlit run main.py
```

### Cloud Deployment
This app can be deployed to:
- Streamlit Cloud
- Heroku
- Railway
- Google Cloud Platform
- AWS

## Project Structure

```
rag-chatbot/
├── main.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── groq_api.txt        # Your API key (not in repo)
```

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: LLM framework
- **ChromaDB**: Vector database for embeddings
- **Groq**: Fast LLM API
- **HuggingFace**: Embeddings model
- **PyPDF**: PDF processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

If you encounter any issues, please open an issue on GitHub.
