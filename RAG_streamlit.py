import streamlit as st
from RAG_utils import ask_question, initialize_rag_system, clear_chat_history
import tiktoken
import re

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Access secret value from Streamlit secrets
api_key = st.secrets["MISTRAL_API_KEY"]

# Initialize RAG system (you can cache this)
@st.cache_resource
def load_rag_system():
    try:
        return initialize_rag_system()
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Load RAG components
rag_components = load_rag_system()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize RAG chat history (separate from display history)
if 'rag_chat_history' not in st.session_state:
    st.session_state.rag_chat_history = []

# Initialize the list in session_state
if "doc_records" not in st.session_state:
    st.session_state.doc_records = []

# Initialize token usage tracking
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = {
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_tokens': 0,
        'requests_count': 0
    }

# Token estimation functions
@st.cache_data
def get_tokenizer():
    """Get a tokenizer for token estimation (using tiktoken as approximation)"""
    try:
        return tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer as approximation
    except:
        return None

def estimate_tokens(text):
    """Estimate token count for given text"""
    if not text:
        return 0
    
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(str(text)))
        except:
            pass
    
    # Fallback: rough estimation (1 token ≈ 4 characters for English)
    return max(1, len(str(text)) // 4)

def update_token_usage(input_text, output_text, context_docs=None):
    """Update token usage statistics"""
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    
    # Add context tokens if available
    context_tokens = 0
    if context_docs:
        context_text = " ".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context_docs])
        context_tokens = estimate_tokens(context_text)
    
    total_input = input_tokens + context_tokens
    
    st.session_state.token_usage['total_input_tokens'] += total_input
    st.session_state.token_usage['total_output_tokens'] += output_tokens
    st.session_state.token_usage['total_tokens'] += (total_input + output_tokens)
    st.session_state.token_usage['requests_count'] += 1
    
    return {
        'input_tokens': total_input,
        'output_tokens': output_tokens,
        'total_tokens': total_input + output_tokens
    }

# Function to display chat messages with different styling for user and chatbot
def display_chat_message(message):
    """
    Display a chat message with styling based on message type
    Args:
        message: Either a LangChain message object (HumanMessage/AIMessage) or dict with 'role' and 'text'
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    # Handle LangChain message objects
    if isinstance(message, HumanMessage):
        role = 'user'
        text = message.content
    elif isinstance(message, AIMessage):
        role = 'chatbot'
        text = message.content
    # Handle dictionary format (for backward compatibility)
    elif isinstance(message, dict):
        role = message.get('role', 'chatbot')
        text = message.get('text', '')
    else:
        # Fallback
        role = 'chatbot'
        text = str(message)
    
    if role == 'user':
        # User message - right aligned, blue styling
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
            <div style="background-color: #007bff; color: white; padding: 12px 16px; 
                        border-radius: 18px 18px 4px 18px; max-width: 70%; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-weight: 500; margin-bottom: 4px;">👤 You</div>
                <div style="white-space: pre-wrap;">{text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:  # chatbot
        # Chatbot message - left aligned, gray styling
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
            <div style="background-color: #f1f3f4; color: #333; padding: 12px 16px; 
                        border-radius: 18px 18px 18px 4px; max-width: 70%; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="font-weight: 500; margin-bottom: 4px;">🤖 RAG ChatBot</div>
                <div style="white-space: pre-wrap;">{text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="RAG Chat Application", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .chat-input-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin-top: 10px;
    }
    
    .stTextArea textarea {
        max-height: 120px !important;
        min-height: 40px !important;
        resize: none !important;
    }
    
    .main > div {
        padding-top: 2rem;
    }
    
    .status-indicator {
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .status-ready {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🤖 RAG-Powered Chat Application")

# System status indicator
if rag_components:
    st.markdown('<div class="status-indicator status-ready">✅ RAG System Ready</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-indicator status-error">❌ RAG System Error</div>', unsafe_allow_html=True)
    st.error("RAG system failed to initialize. Please check your configuration.")

# Clear chat button (positioned above chat screen)
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🗑️ Clear Chat", key="clear_btn", help="Clear all chat messages"):
        st.session_state.chat_history = []
        st.session_state.rag_chat_history = clear_chat_history()
        st.session_state.doc_records = []
        # Reset token usage
        st.session_state.token_usage = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'requests_count': 0
        }
        st.rerun()

with col2:
    # Show chat history length
    st.markdown(f"**Messages: {len(st.session_state.chat_history)}**")

# Chat screen container
st.markdown("**💬 Chat Messages:**")

if st.session_state.chat_history:
    # Display all messages in chat history
    for message in st.session_state.chat_history:
        display_chat_message(message)
else:
    st.info("📝 No messages yet. Start typing in the chat box below to begin the conversation!")

# Create two columns for input and button
input_col, button_col = st.columns([5, 1])

with input_col:
    # Text area with fixed dimensions
    user_input = st.text_area(
        "Ask a question about the documentation:",
        height=100,
        max_chars=1000,
        placeholder="Enter your question here (max 5 lines)...",
        key="chat_input"
    )

with button_col:
    st.write("")  # Add some spacing
    st.write("")  # Add more spacing to align with text area
    send_button = st.button("📤 Send", key="send_btn", help="Click to send message")

st.markdown('</div>', unsafe_allow_html=True)

# Handle message sending
if send_button and user_input.strip():
    # Check if RAG system is available
    if not rag_components:
        st.error("❌ RAG system is not available. Cannot process your question.")
    else:
        # Check if message has more than 5 lines
        lines = user_input.split('\n')
        if len(lines) > 5:
            st.error("⚠️ Message cannot exceed 5 lines. Please shorten your message.")
        else:
            # Add user message to chat history (as HumanMessage)
            from langchain_core.messages import HumanMessage, AIMessage
            
            user_message = HumanMessage(content=user_input.strip())
            st.session_state.chat_history.append(user_message)
            
            # Show processing indicator
            with st.spinner('🔍 Processing your question...'):
                try:
                    # Get RAG response
                    result = ask_question(
                        vector_db=rag_components["vector_db"],
                        input_question=user_input.strip(),
                        chat_history=st.session_state.rag_chat_history,
                        llm=rag_components["llm"],
                        contextualized_prompt_for_retriever=rag_components["contextualized_prompt"],
                        qa_prompt=rag_components["qa_prompt"]
                    )
                    
                    # Update RAG chat history
                    st.session_state.rag_chat_history = result["chat_history"]
                    
                    # Get the response and create AIMessage
                    chatbot_response = result["answer"]
                    retrieved_docs = result.get("context", [])

                    source_map = {}
                    for doc in retrieved_docs:
                        source = doc.metadata.get("source")
                        page = doc.metadata.get("page_nos")

                        if source and page:
                            if source not in source_map:
                                source_map[source] = []
                            if page not in source_map[source]:
                                source_map[source].append(page)

                    # Format as a list of dicts to store
                    sources_list = [{"source": src, "page_nos": pages} for src, pages in source_map.items()]
                    st.session_state.doc_records.append(sources_list)

                    # Update token usage tracking
                    token_info = update_token_usage(
                        input_text=user_input.strip(),
                        output_text=chatbot_response,
                        context_docs=retrieved_docs
                    )
                    
                    ai_message = AIMessage(
                        content=chatbot_response
                    )
                    
                    # Add chatbot response to display chat history
                    st.session_state.chat_history.append(ai_message)
                    
                except Exception as e:
                    # Handle errors
                    error_message = f"Sorry, I encountered an error while processing your question: {str(e)}"
                    ai_error_message = AIMessage(content=error_message)
                    st.session_state.chat_history.append(ai_error_message)
            
            st.rerun()

elif send_button and not user_input.strip():
    st.warning("⚠️ Please enter a message before sending.")

# Sidebar with additional information
with st.sidebar:
    st.header("📊 Chat Statistics")
    
    # Basic chat stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.chat_history))
        st.metric("Requests", st.session_state.token_usage['requests_count'])
    with col2:
        st.metric("RAG History", len(st.session_state.rag_chat_history))

    
        
    # Token usage statistics
    st.header("🪙 Token Usage")
    
    # Display token metrics
    st.metric(
        "Total Tokens", 
        f"{st.session_state.token_usage['total_tokens']:,}",
        help="Estimated total tokens used (input + output)"
    )
    
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.metric(
    #         "Input Tokens", 
    #         f"{st.session_state.token_usage['total_input_tokens']:,}",
    #         help="Tokens from user questions + context"
    #     )
    # with col2:
    #     st.metric(
    #         "Output Tokens", 
    #         f"{st.session_state.token_usage['total_output_tokens']:,}",
    #         help="Tokens from AI responses"
    #     )
    
    # Average tokens per request
    if st.session_state.token_usage['requests_count'] > 0:
        avg_tokens = st.session_state.token_usage['total_tokens'] / st.session_state.token_usage['requests_count']
        st.metric("Avg Tokens/Request", f"{avg_tokens:.1f}")
    
    # Token usage chart
    # if st.session_state.token_usage['total_tokens'] > 0:
    #     st.subheader("Token Distribution")
    #     input_pct = (st.session_state.token_usage['total_input_tokens'] / st.session_state.token_usage['total_tokens']) * 100
    #     output_pct = (st.session_state.token_usage['total_output_tokens'] / st.session_state.token_usage['total_tokens']) * 100
        
    #     st.progress(input_pct / 100, text=f"Input: {input_pct:.1f}%")
    #     st.progress(output_pct / 100, text=f"Output: {output_pct:.1f}%")
    
    st.header("⚙️ System Status")
    if rag_components:
        st.success("✅ Vector DB: Connected")
        st.success("✅ LLM: Ready")
        st.success("✅ Embeddings: Loaded")
        st.info("🪙 Token tracking: Estimated")
    else:
        st.error("❌ System: Not Ready")
    
    # Debug section (expandable)
    # with st.expander("🔧 Debug Info"):
    #     if st.session_state.chat_history:
    #         st.write("**Last Messages:**")
    #         if len(st.session_state.chat_history) >= 2:
    #             last_message = st.session_state.chat_history[-1]
    #             from langchain_core.messages import HumanMessage, AIMessage
    #             if isinstance(last_message, AIMessage):
    #                 st.text_area("Last AI Response", last_message.content, height=100, disabled=True)
    #             elif isinstance(last_message, HumanMessage):
    #                 st.text_area("Last User Message", last_message.content, height=50, disabled=True)

    with st.expander("Source:"):
        if st.session_state.chat_history:
            if len(st.session_state.chat_history) >= 2:
                latest_sources = st.session_state.doc_records[-1]
                if latest_sources:
                    references = []
                    for entry in latest_sources:
                        pages_str = ", ".join(map(str, entry["page_nos"]))
                        page_label = "page" if len(entry["page_nos"]) == 1 else "pages"
                        references.append(f"{entry['source']} ({page_label} {pages_str})")
                    
                    reference_text = "You can refer to documents on " + " and ".join(references) + "."
                    st.write(reference_text)
                else:
                    st.text_area("No sources found for the latest chat.", height=50, disabled=True)

        # Token estimation info
        # st.write("**Token Estimation:**")
        # st.caption("Using tiktoken (GPT-4 tokenizer) as approximation for Ollama models")
        # st.caption("Actual token usage may vary depending on the model")

# Add some information about the app
with st.expander("ℹ️ RAG Chat Features"):
    st.markdown("""
    **Features of this RAG-powered chat application:**
    - 🤖 **RAG Integration**: Powered by retrieval-augmented generation
    - 📚 **Document Search**: Searches through your knowledge base
    - 🧠 **Context Awareness**: Maintains conversation context
    - 📝 **Fixed-size input**: Text area with maximum 5 lines
    - 📤 **Send button**: Click to send your message
    - 💬 **Chat history**: Messages stack from oldest to newest 
    - 🗑️ **Clear chat**: Remove all messages at once
    - 💾 **Persistent state**: Messages are kept during your session unless cleared
    - ⚡ **Real-time processing**: Get answers from your documents
    Noted: This is only a personal project, hence the llm and embedding models was the lowest cost models available at the time of development.
    """)

# Footer
st.markdown("---")
st.markdown("*RAG-Powered Chat Application built with Streamlit* 🚀")