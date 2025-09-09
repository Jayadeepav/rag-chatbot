import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from src.document_processor import process_document, chunk_text
from src.vector_store import get_vector_store, add_to_vector_store
from src.rag_pipeline import query_documents

# Page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()

# Helper: format timestamp
def format_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Sidebar: Document upload and management
with st.sidebar:
    st.header("📁 Document Management")

    # Model check and download info
    embedding_model_path = "models/all-MiniLM-L6-v2"
    if not os.path.exists(embedding_model_path):
        st.warning("Embedding model not found. Please run download_model.py")

    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / uploaded_file.name

        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("Process and Add to Knowledge Base"):
            with st.spinner("Processing document and adding to knowledge base..."):
                try:
                    text = process_document(file_path)
                    chunks = chunk_text(text)

                    # Add to vector store
                    add_to_vector_store(chunks, uploaded_file.name)

                    # Get vector store instance and show document count
                    vs = get_vector_store()
                    doc_count = vs.get_count()
                    st.success(f"Added {len(chunks)} chunks to knowledge base!")
                    st.info(f"Total documents in vector store: {doc_count}")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    st.markdown("---")
    st.header("🗄️ Database Management")
    if st.button("Show Database Info"):
        vs = st.session_state.vector_store
        count = vs.get_count()
        st.info(f"Documents in database: {count}")

    if st.button("Clear Database"):
        if st.checkbox("Confirm deletion"):
            vs = st.session_state.vector_store
            vs.clear()
            st.success("Database cleared!")

    st.markdown("---")
    st.markdown("### 📖 How to use:")
    st.markdown("""
    1. Upload a document (PDF or TXT)  
    2. Click 'Process and Add to Knowledge Base'  
    3. Ask questions about your documents below  
    4. View confidence scores and retrieved context  
    """)

# Main app layout
st.title("📚 RAG Chatbot")

# Chat input and controls in columns
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Ask a question about your documents:", key="query_input")
with col2:
    top_k = st.slider("Context chunks:", 1, 5, 3)
    get_answer = st.button("Get Answer")

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

# Process query
if get_answer and query.strip():
    with st.spinner("Searching and generating answer..."):
        try:
            results = query_documents(query, top_k=top_k)

            # Save chat history with timestamp
            st.session_state.chat_history.append({
                "user": query,
                "bot": results["answer"],
                "timestamp": format_timestamp()
            })

            # Display answer and confidence
            st.subheader("Answer:")
            st.markdown(f"**{results['answer']}**")

            if results.get("confidence") is not None:
                confidence_percent = results["confidence"] * 100
                st.metric("Confidence Score", f"{confidence_percent:.1f}%")

            # Similarity scores
            with st.expander("View Similarity Scores"):
                similarities = results.get("similarities", [])
                if not isinstance(similarities, list):
                    similarities = [similarities] if similarities is not None else []

                for i, sim in enumerate(similarities):
                    sim_percent = sim * 100
                    st.write(f"Context {i+1}: {sim_percent:.1f}% similar")
                    st.progress(float(sim))

            # Retrieved context with optional highlighting
            with st.expander("View Retrieved Context"):
                for i, context in enumerate(results.get("contexts", [])):
                    st.markdown(f"**Context {i+1}:**")
                    # Show first 500 chars with ellipsis if long
                    display_text = context if len(context) <= 500 else context[:500] + "..."
                    st.write(display_text)
                    st.divider()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure you have documents in the knowledge base and models are downloaded.")

elif get_answer and not query.strip():
    st.warning("Please enter a question before clicking 'Get Answer'.")

# Display chat history nicely with timestamps
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.chat_message("user").write(f"{chat['user']}  \n*{chat['timestamp']}*")
        st.chat_message("assistant").write(chat["bot"])

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")