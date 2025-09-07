import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings
from pathlib import Path

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(
    page_title="Document Processor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Processing System")
st.markdown("""
Upload PDF or TXT documents, split into chunks, generate embeddings, and store in ChromaDB.
""")

# -------------------------------
# Create data directory if missing
# -------------------------------
Path("./data").mkdir(exist_ok=True)

# -------------------------------
# ChromaDB Client (Persistent)
# -------------------------------
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

# -------------------------------
# Embedding Model
# -------------------------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

# -------------------------------
# Get or Create Collection
# -------------------------------
def get_collection():
    client = get_chroma_client()
    try:
        collection = client.get_collection(name="documents")
    except ValueError:
        collection = client.create_collection(name="documents")
    return collection

# -------------------------------
# File Upload Section
# -------------------------------
st.header("Step 1: Upload Document")
uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"],
    help="Select a PDF or text file to process"
)

if uploaded_file:
    file_details = {
        "Filename": uploaded_file.name,
        "Size": f"{uploaded_file.size / 1024:.2f} KB",
        "Type": uploaded_file.type
    }
    st.write(file_details)

    # -------------------------------
    # Text Extraction
    # -------------------------------
    st.header("Step 2: Text Extraction")
    with st.spinner("Extracting text..."):
        text = ""
        if uploaded_file.type == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                st.success("PDF text extracted successfully!")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        else:
            try:
                text = uploaded_file.getvalue().decode("utf-8")
                st.success("Text file extracted successfully!")
            except Exception as e:
                st.error(f"Error reading text file: {e}")

    if text:
        with st.expander("Preview Extracted Text"):
            st.text(text[:1000] + "..." if len(text) > 1000 else text)

        # -------------------------------
        # Text Chunking
        # -------------------------------
        st.header("Step 3: Text Chunking")
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        chunk_size = st.slider("Chunk Size", 100, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)

        if st.button("Split Text into Chunks"):
            with st.spinner("Splitting text..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                st.success(f"Text split into {len(chunks)} chunks!")
                st.session_state.chunks = chunks

                # Show sample
                for i, chunk in enumerate(chunks[:3]):
                    st.markdown(f"*Chunk {i+1}* (Length: {len(chunk)})")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.divider()

# -------------------------------
# Database Management
# -------------------------------
st.header("Database Management")
collection = get_collection()
if st.button("Show Database Stats"):
    try:
        data = collection.get()
        st.info(f"Database contains {len(data['ids'])} document chunks")
        with st.expander("Sample Entries"):
            for i in range(min(3, len(data['ids']))):
                st.markdown(f"*Entry {i+1}*")
                st.text(f"ID: {data['ids'][i]}")
                st.text(f"Document: {data['documents'][i][:100]}...")
                st.divider()
    except Exception as e:
        st.error(f"Error fetching database info: {e}")

if st.button("Clear Database"):
    try:
        client = get_chroma_client()
        client.delete_collection(name="documents")
        client.create_collection(name="documents")
        st.success("Database cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing database: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("### Next Steps")

# -------------------------------
# Step 4: Generate Embeddings & Store
# -------------------------------
if 'chunks' in st.session_state and st.session_state['chunks']:
    st.header("Step 4: Generate Embeddings & Store")
    if st.button("Generate & Store"):
        with st.spinner("Generating embeddings and storing into DB..."):
            try:
                embedder = get_embedder()
                chunks = st.session_state['chunks']
                collection = get_collection()
                base_name = uploaded_file.name if uploaded_file else "uploaded"
                ids = [f"{base_name}_{i}" for i in range(len(chunks))]
                existing_ids = collection.get()['ids'] if collection.count() > 0 else []
                new_chunks = []
                new_embeddings = []
                new_ids = []
                for i, chunk_id in enumerate(ids):
                    if chunk_id not in existing_ids:
                        new_chunks.append(chunks[i])
                        embedding = embedder.encode(chunks[i]).tolist()
                        new_embeddings.append(embedding)
                        new_ids.append(chunk_id)
                if new_chunks:
                    collection.add(documents=new_chunks, embeddings=new_embeddings, ids=new_ids)
                    st.success(f"Stored {len(new_chunks)} new chunks in the database!")
                else:
                    st.info("All chunks already exist in the database. No duplicates stored.")
            except Exception as e:
                st.error(f"Error during embedding: {e}")
else:
    st.header("Step 4: Generate Embeddings & Store")
    st.info("No chunks found. Please split text into chunks in Step 3 first.")

# -------------------------------
# Step 5: Retrieval + Question Answering
# -------------------------------
st.header("Step 5: Ask Questions with RAG")

query = st.text_input(
    "Enter your question",
    placeholder="e.g. What is explained about AI in the document?",
    key="rag_query"  # Unique key to avoid DuplicateWidgetID error
)

if query:
    try:
        with st.spinner("Searching and generating answer..."):
            collection = get_collection()
            embedder = get_embedder()
            query_embedding = embedder.encode([query]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=3)
            retrieved_context = "\n".join(results["documents"][0])

            # Use transformers pipeline for question answering
            qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

            result = qa_pipeline({
                "context": retrieved_context,
                "question": query
            })

            answer = result['answer']

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Retrieved Context"):
                st.text(retrieved_context)

    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")