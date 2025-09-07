import streamlit as st
import os
from pathlib import Path
from src.document_processor import process_document, chunk_text
from src.vector_store import get_vector_store, add_to_vector_store
from src.rag_pipeline import query_documents

# Streamlit app setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot")

# Check if models are available
def check_models():
    """Check if required models are available"""
    embedding_model_path = "models/all-MiniLM-L6-v2"
    
    warnings = []
    if not os.path.exists(embedding_model_path):
        warnings.append("Embedding model not found. Please run download_model.py")
    
    return warnings

# Initialize
model_warnings = check_models()
for warning in model_warnings:
    st.warning(warning)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file:
        # Save uploaded file
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner("Processing document..."):
            try:
                text = process_document(file_path)
                chunks = chunk_text(text)
                
                if st.button("Add to Knowledge Base"):
                    add_to_vector_store(chunks, uploaded_file.name)
                    st.success(f"Added {len(chunks)} chunks to knowledge base!")
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    # In the sidebar section of app.py, add:
if st.sidebar.button("🛠️ Debug - Check Documents"):
    vs = st.session_state.vector_store
    count = vs.get_count()
    st.sidebar.info(f"Documents in database: {count}")
    
    if count > 0:
        st.sidebar.write("Sample document content:")
        for i, doc in enumerate(vs.documents[:2]):  # Show first 2 docs
            st.sidebar.text_area(f"Document {i+1}", doc[:200] + "..." if len(doc) > 200 else doc, height=100)
    else:
        st.sidebar.warning("No documents found in database!")

# Main chat interface
st.header("💬 Chat with Your Documents")

query = st.text_input("Ask a question about your documents:", key="query_input")
top_k = st.slider("Number of context chunks:", 1, 5, 3)

if query and st.button("Get Answer"):
    with st.spinner("Searching and generating answer..."):
        try:
            results = query_documents(query, top_k=top_k)
            
            st.subheader("Answer:")
            st.write(results["answer"])
            
            # Confidence display
            if "confidence" in results:
                confidence_percent = results["confidence"] * 100
                st.metric("Confidence Score", f"{confidence_percent:.1f}%")
                
                # Show similarity scores for each context
                with st.expander("View Similarity Scores"):
                    similarities = results.get("similarities", [])
                    # Ensure similarities is a list, not a single value
                    if not isinstance(similarities, list):
                        similarities = [similarities] if similarities is not None else []
                    
                    for i, context in enumerate(results["contexts"]):
                        if i < len(similarities):
                            similarity = similarities[i]
                            sim_percent = similarity * 100
                            st.write(f"**Context {i+1}**: {sim_percent:.1f}% similar")
                            st.progress(float(similarity))  # Ensure it's a float
                        else:
                            st.write(f"**Context {i+1}**: Similarity score not available")
            
            with st.expander("View Retrieved Context"):
                for i, context in enumerate(results["contexts"]):
                    st.markdown(f"**Context {i+1}:**")
                    st.write(context[:500] + "..." if len(context) > 500 else context)
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure you have documents in the knowledge base and models are downloaded.")

# Database info
if st.sidebar.button("Show Database Info"):
    vs = st.session_state.vector_store
    count = vs.get_count()
    st.sidebar.info(f"Documents in database: {count}")

if st.sidebar.button("Clear Database"):
    if st.sidebar.checkbox("Confirm deletion"):
        vs = st.session_state.vector_store
        vs.clear()
        st.sidebar.success("Database cleared!")

# Footer with instructions
st.markdown("---")
st.markdown("### 📖 How to use:")
st.markdown("""
1. **Upload a document** (PDF or TXT) using the sidebar
2. **Click 'Add to Knowledge Base'** to process and store it
3. **Ask questions** about your documents in the chat interface
4. **View confidence scores** and retrieved context for each answer
""")