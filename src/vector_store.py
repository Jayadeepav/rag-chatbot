import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.documents = []
        self.embeddings = None
        self._load_from_disk()
    
    def _load_from_disk(self):
        """Load FAISS index and documents from disk if they exist"""
        storage_dir = Path("./storage/faiss_index")
        storage_dir.mkdir(exist_ok=True, parents=True)
        
        index_path = storage_dir / "index.npy"
        docs_path = storage_dir / "documents.pkl"
        
        if index_path.exists() and docs_path.exists():
            try:
                self.embeddings = np.load(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Loaded {len(self.documents)} documents from storage")
            except Exception as e:
                print(f"Error loading from disk: {e}")
                self.embeddings = np.array([])
                self.documents = []
        else:
            self.embeddings = np.array([])
            self.documents = []
    
    def _save_to_disk(self):
        """Save FAISS index and documents to disk"""
        storage_dir = Path("./storage/faiss_index")
        storage_dir.mkdir(exist_ok=True, parents=True)
        
        if self.embeddings is not None and len(self.embeddings) > 0:
            np.save(storage_dir / "index.npy", self.embeddings)
            with open(storage_dir / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"Saved {len(self.documents)} documents to storage")
    
    def add_documents(self, documents, metadata=None, ids=None):
        """Add documents to vector store"""
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Generate embeddings
        new_embeddings = self.embedder.encode(documents)
        
        # Add to existing embeddings
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Add to documents list
        self.documents.extend(documents)
        
        # Save to disk
        self._save_to_disk()
        
        print(f"Total documents in store: {len(self.documents)}")
    
    def search(self, query, top_k=3):
        """Search for similar documents using cosine similarity"""
        if len(self.documents) == 0:
            return {"documents": [[]], "distances": [[]]}
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        
        # Ensure embeddings are 2D
        if len(self.embeddings.shape) == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        
        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        doc_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Calculate similarities (cosine similarity)
        similarities = np.dot(query_norm, doc_norms.T)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = {
            "documents": [[self.documents[i] for i in top_indices]],
            "distances": [similarities[i] for i in top_indices]
        }
        
        return results
    
    def get_count(self):
        """Get number of documents in collection"""
        return len(self.documents)
    
    def clear(self):
        """Clear the collection"""
        self.documents = []
        self.embeddings = np.array([])
        
        # Remove files from disk
        storage_dir = Path("./storage/faiss_index")
        for file in storage_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass
        print("Vector store cleared")

# Global instance
_vector_store_instance = None

def get_vector_store():
    """Get or create vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance

def add_to_vector_store(chunks, source_name):
    """Add chunks to vector store"""
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)