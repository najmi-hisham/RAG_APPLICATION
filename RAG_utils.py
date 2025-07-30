# Cell 1: Import required libraries
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Cell 2: Chat history utility functions (no global initialization)
def add_to_chat_history(chat_history, question, response):
    """Add question and response pair to chat history using extend method"""
    chat_history.extend([
        HumanMessage(content=question),       
        AIMessage(content=response)
    ])
    return chat_history

def clear_chat_history():
    """Clear the chat history - returns empty list"""
    return []

def display_chat_history(chat_history):
    """Display formatted chat history"""
    if not chat_history:
        print("No chat history yet.")
        return
    
    print("=== Chat History ===")
    for i, msg in enumerate(chat_history):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{i+1}. {role}: {msg.content}")
    print("==================")

def test_contextualization(input_question, chat_history, context_prompt_chain):
    """Test how the LLM rephrases questions given chat history"""
    if chat_history is None or len(chat_history) == 0:
        return input_question
    
    result = context_prompt_chain.invoke({
        "chat_history": chat_history,
        "input": input_question
    })
    
    print(f"Original: {input_question}")
    print(f"Rephrased: {result}")
    print("-" * 50)
    return result

def initialize_vector_db(embedding_model=None):
    """Initialize the vector database with persisted data"""
    
    # Load the embedding model (must match the one used during creation)
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    # Initialize Chroma with the persisted data
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    return vector_db

def load_vector_db(embedding_model=None):
    """Load the vector database from the persisted directory"""
    
    # Load the embedding model (must match the one used during creation)
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    # Initialize Chroma with the persisted data
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    return vector_db

# Cell 3: Initialize the LLM model 
def initialize_llm():
    from langchain_mistralai import ChatMistralAI
    
    # Create the LLM object
    llm = ChatMistralAI(
        model="mistral-small-latest",  # or mistral-medium-latest, mistral-large-latest
        temperature=0.7,
        max_retries=2
    )
    return llm

# Cell 4: Context prompt for history-aware retrieval
def get_contextualized_prompt():
    """Get the contextualized prompt for retriever"""
    context_prompt = """Given a latest question and past chat history, 
                        rephrase the question to make it more understandable with only a single sentence. 
                        For example, if the past question taling about a specific tools and user
                        ask about it using 'it','the tool' etc that point to the tool, please rephrase it using it full name to
                        make it like a standalone question.
                        the rephrased question should 
                        DO NOT ANSWER the question!! Just rephrase it to make it like a standalone 
                        question without chat history."""

    return ChatPromptTemplate.from_messages([
        ("system", context_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

# Cell 5: Main prompt template for RAG
def get_qa_prompt():
    """Get the QA prompt template"""
    template = """You are a helpful chatbot that help people especially developer understanding about a documnetation.

Context from knowledge base:
{context}

Question: {input}

Instructions:
1. First, check if the context directly answers the question
2. If the context contains relevant information, use it as your main source
3. If the question requires connecting the context to other topics (like dietary preferences, cultural tastes, or comparisons), use your general knowledge to make those connections
4. Always cite sources when using information from the context: [source: document-name]
5. Be honest if information is not in the context but you're using general knowledge
6. Provide practical, helpful recommendations when possible

Answer in a friendly, informative tone:"""

    return PromptTemplate(template=template, input_variables=["context", "input"])

# Cell 6: Create the complete RAG chain
def create_rag_chain(vector_db, llm, contextualized_prompt_for_retriever, qa_prompt):
    """
    Create the complete RAG chain with history-aware retrieval
    
    Args:
        vector_db: Your vector store instance
        llm: Language model instance
        contextualized_prompt_for_retriever: Contextualized prompt
        qa_prompt: QA prompt template
        
    Returns:
        Complete RAG chain that can handle questions with chat history
    """
    # Create retriever
    retriever = vector_db.as_retriever(
        search_type="similarity",  # Default ("similarity" / "mmr" / "similarity_score_threshold")
        search_kwargs={"k": 5}     # Number of docs to return (adjust as needed)
    )
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualized_prompt_for_retriever
    )
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    return rag_chain

# Cell 7: Main chat function
def ask_question(vector_db, input_question, chat_history, llm=None, contextualized_prompt_for_retriever=None, qa_prompt=None):
    """
    Ask a question about documentation with chat history context
    
    Args:
        vector_db: Your vector database instance
        input_question: User's question 
        chat_history: Current chat history (list of messages)
        llm: Language model instance (optional, will initialize if None)
        contextualized_prompt_for_retriever: Contextualized prompt (optional)
        qa_prompt: QA prompt (optional)
        
    Returns:
        Dictionary with response and updated chat history
    """
    # Initialize components if not provided
    if llm is None:
        llm = initialize_llm()
    
    if contextualized_prompt_for_retriever is None:
        contextualized_prompt_for_retriever = get_contextualized_prompt()
    
    if qa_prompt is None:
        qa_prompt = get_qa_prompt()
    
    # Create context prompt chain for testing
    context_prompt_chain = contextualized_prompt_for_retriever | llm
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_db, llm, contextualized_prompt_for_retriever, qa_prompt)
    
    # Get response using current chat history
    result = rag_chain.invoke({
        "input": input_question,
        "chat_history": chat_history
    })
    
    response = result["answer"]
    retrieved_docs = result.get("context", [])
    
    # Create updated chat history
    updated_chat_history = chat_history.copy()
    updated_chat_history = add_to_chat_history(updated_chat_history, input_question, response)

    unique_page_nos = set(doc.metadata['page_nos'] for doc in retrieved_docs)
    return {
        "question": input_question,
        "rephrased_question": test_contextualization(input_question, chat_history, context_prompt_chain),
        "answer": response,
        "context": retrieved_docs,
        "chat_history": updated_chat_history,
        "chat_history_length": len(updated_chat_history)
    }

# Convenience function to initialize all components
def initialize_rag_system():
    """Initialize all RAG system components"""
    llm = initialize_llm()
    vector_db = load_vector_db()
    contextualized_prompt = get_contextualized_prompt()
    qa_prompt = get_qa_prompt()
    
    return {
        "llm": llm,
        "vector_db": vector_db,
        "contextualized_prompt": contextualized_prompt,
        "qa_prompt": qa_prompt
    }