import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
import os

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
    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./data/chroma_db"
    ))

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

                # Display sample chunks
                for i, chunk in enumerate(chunks[:3]):
                    st.markdown(f"**Chunk {i+1}** (Length: {len(chunk)})")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.divider()

        # -------------------------------
        # Embedding Generation & Storage
        # -------------------------------
        if 'chunks' in st.session_state:
            st.header("Step 4: Generate Embeddings & Store")
            if st.button("Generate & Store"):
                with st.spinner("Generating embeddings..."):
                    embedder = get_embedder()
                    collection = get_collection()
                    chunks = st.session_state.chunks
                    embeddings = embedder.encode(chunks)

                    # Generate unique IDs and check for duplicates
                    ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
                    existing_ids = collection.get()['ids'] if collection.count() > 0 else []
                    new_chunks = []
                    new_embeddings = []
                    new_ids = []
                    for i, chunk_id in enumerate(ids):
                        if chunk_id not in existing_ids:
                            new_chunks.append(chunks[i])
                            new_embeddings.append(embeddings[i].tolist())
                            new_ids.append(chunk_id)

                    if new_chunks:
                        collection.add(
                            documents=new_chunks,
                            embeddings=new_embeddings,
                            ids=new_ids
                        )
                        st.success(f"Stored {len(new_chunks)} new chunks in the database!")
                    else:
                        st.info("All chunks already exist in the database. No duplicates stored.")

                    # Display stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks Processed", len(chunks))
                    col2.metric("Embedding Dim", len(embeddings[0]))
                    col3.metric("DB Entries", collection.count())

                    with st.expander("Sample Stored Data"):
                        sample = collection.get(ids=new_ids[:3])
                        for i, doc in enumerate(sample['documents']):
                            st.markdown(f"**Entry {i+1}**")
                            st.text(f"{doc[:100]}..." if len(doc) > 100 else doc)
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
                st.markdown(f"**Entry {i+1}**")
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

