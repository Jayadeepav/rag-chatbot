# ingest.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

import os
import pickle


# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_FOLDER = os.path.join(BASE_DIR, "data")       # data folder in root
STORAGE_FOLDER = os.path.join(BASE_DIR, "storage") # store outputs in /storage

# Ensure storage folder exists
os.makedirs(STORAGE_FOLDER, exist_ok=True)


# Load documents

def load_documents(data_folder=DATA_FOLDER):
    docs = []
    for file in os.listdir(data_folder):
        path = os.path.join(data_folder, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())
    return docs


# Split into chunks

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


#  Generate embeddings

def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(texts)
    return embeddings, embed_model

#  Store in FAISS

def store_in_faiss(texts, embeddings, embed_model):
    text_embeddings = list(zip(texts, embeddings))
    db = FAISS.from_embeddings(text_embeddings, embed_model)

    # Save inside /storage/faiss_index
    index_path = os.path.join(STORAGE_FOLDER, "faiss_index")
    db.save_local(index_path)
    print(f" Embeddings stored in FAISS at {index_path}")


#  Save chunks and embeddings as proof

def save_proof(texts, embeddings):
    chunks_path = os.path.join(STORAGE_FOLDER, "chunks.pkl")
    embeddings_path = os.path.join(STORAGE_FOLDER, "embeddings.pkl")

    with open(chunks_path, "wb") as f:
        pickle.dump(texts, f)
    print(f" Chunks saved to {chunks_path}")

    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f" Embeddings saved to {embeddings_path}")


#  View chunks and embeddings

def view_proof():
    chunks_path = os.path.join(STORAGE_FOLDER, "chunks.pkl")
    embeddings_path = os.path.join(STORAGE_FOLDER, "embeddings.pkl")

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print("🔹 First 5 chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i+1}:\n{chunk}\n")

    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    print("🔹 First 5 embeddings (showing first 10 values each):")
    for i, emb in enumerate(embeddings[:5]):
        print(f"Embedding {i+1} (length {len(emb)}): {emb[:10]}...\n")


# Main Script

if __name__ == "__main__":
    print(" Loading documents...")
    documents = load_documents()

    print(" Splitting into chunks...")
    chunks = split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    print(f"Generated {len(texts)} chunks")

    print(" Generating embeddings...")
    embeddings, embed_model = generate_embeddings(texts)
    print(f" Generated embeddings shape: {len(embeddings)} x {len(embeddings[0])}")

    save_proof(texts, embeddings)

    print(" Storing in FAISS...")
    store_in_faiss(texts, embeddings, embed_model)

    index_path = os.path.join(STORAGE_FOLDER, "faiss_index")
    new_db = FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
    print(f" Reloaded FAISS index with {new_db.index.ntotal} vectors")

    print("\n Viewing saved chunks and embeddings:")
    view_proof()
