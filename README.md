# RAG Chatbot - Simple Retrieval-Augmented Generation Demo

A minimal implementation of a Retrieval-Augmented Generation (RAG) chatbot that demonstrates how RAG systems work by combining document retrieval with language model generation.

## 🎯 Purpose

This project serves as an educational showcase of RAG (Retrieval-Augmented Generation) technology, demonstrating how to:
- Extract and embed content from documents
- Store embeddings in a vector database
- Retrieve relevant context based on user queries
- Generate informed responses using retrieved context

## 🏗️ Architecture

```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

The system follows the classic RAG pipeline:
1. **Document Processing**: PDF content is processed using Docling
2. **Embedding Creation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in ChromaDB
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Relevant document chunks are retrieved
6. **Response Generation**: LLM generates answers using retrieved context

## 🔧 Tech Stack

- **Document Processing**: Docling (for PDF parsing and content extraction)
- **Vector Database**: ChromaDB (for embedding storage and similarity search)
- **Embeddings**: [Your embedding model - e.g., OpenAI, HuggingFace, etc.]
- **LLM**: [Your language model - e.g., OpenAI GPT, local model, etc.]
- **Backend**: [Your framework - e.g., Python/FastAPI, Flask, etc.]

## 📁 Project Structure

```
rag-chatbot/
├── src/
│   ├── document_processor.py    # PDF processing with Docling
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # ChromaDB operations
│   ├── retriever.py            # Context retrieval logic
│   ├── chatbot.py              # Main chatbot logic
│   └── main.py                 # Application entry point
├── data/
│   └── docling_docs.pdf        # Source document (Docling documentation)
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Usage

1. **Initialize the system**
   ```bash
   python src/main.py --setup
   ```
   This will:
   - Process the Docling documentation PDF
   - Generate embeddings for document chunks
   - Store embeddings in ChromaDB

2. **Start the chatbot**
   ```bash
   python src/main.py
   ```

3. **Ask questions about Docling**
   ```
   > What is Docling?
   > How do I install Docling?
   > What file formats does Docling support?
   ```

## 🔍 How RAG Works (Demo Flow)

This implementation demonstrates the core RAG concepts:

1. **Document Ingestion**: The Docling documentation PDF is processed and split into meaningful chunks
2. **Vectorization**: Each chunk is converted to a high-dimensional vector representation
3. **Storage**: Vectors are stored in ChromaDB with metadata for efficient retrieval
4. **Query Processing**: When you ask a question, it's also converted to a vector
5. **Similarity Search**: The system finds the most relevant document chunks
6. **Context Assembly**: Retrieved chunks are combined to form context
7. **Generation**: The LLM generates a response using both your question and the retrieved context

## 📊 Example Interaction

```
User: "How do I extract text from a PDF using Docling?"

System Process:
1. Convert query to embedding
2. Search ChromaDB for similar content
3. Retrieve top-k relevant chunks about PDF text extraction
4. Send query + context to LLM
5. Generate contextually accurate response

Response: "To extract text from a PDF using Docling, you can use the DocumentConverter class..."
```

## ⚙️ Configuration

Key settings in `config.py`:

- `CHUNK_SIZE`: Size of document chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `TOP_K_RESULTS`: Number of similar chunks to retrieve (default: 3)
- `EMBEDDING_MODEL`: Model used for embeddings
- `LLM_MODEL`: Language model for response generation

## 🎯 Key Features

- **Simple Setup**: Single PDF source for easy understanding
- **Educational**: Clear demonstration of RAG principles
- **Extensible**: Easy to modify for different documents or use cases
- **Local Storage**: Uses ChromaDB for local vector storage
- **Contextual Responses**: Answers are grounded in the source document

## 🔄 Limitations & Future Improvements

**Current Limitations:**
- Single document source (by design for simplicity)
- No conversation memory
- Basic chunking strategy

**Potential Enhancements:**
- Multi-document support
- Conversational context
- Advanced chunking strategies
- Hybrid search (semantic + keyword)
- Response citations

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork and experiment with different configurations
- Try different embedding models
- Implement additional features
- Share your learnings!

## 📝 License

[Your chosen license]

## 🙏 Acknowledgments

- [Docling](https://github.com/DS4SD/docling) for document processing
- [ChromaDB](https://www.trychroma.com/) for vector storage
- The RAG research community for pioneering this approach

---

**Note**: This project is designed for educational purposes to demonstrate RAG concepts. For production use, consider additional features like error handling, monitoring, and security measures.
