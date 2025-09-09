import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

class VectorStore:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        # Load embedder
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        # documents list and embeddings matrix (N x dim)
        self.documents = []  # list of dicts with "page_content"
        self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        self._storage_dir = Path("./storage/faiss_index")
        self._storage_dir.mkdir(exist_ok=True, parents=True)
        self._index_path = self._storage_dir / "index.npy"
        self._docs_path = self._storage_dir / "documents.pkl"
        self._load_from_disk()

    def _load_from_disk(self):
        """Load embeddings and documents if they exist on disk."""
        try:
            if self._index_path.exists() and self._docs_path.exists():
                emb = np.load(self._index_path)
                with open(self._docs_path, "rb") as f:
                    docs = pickle.load(f)
                # Validate shape
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                if emb.shape[1] != self.dim:
                    print("Warning: saved embeddings dimension != model dimension. Clearing loaded embeddings.")
                    self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
                    self.documents = []
                    return
                self.embeddings = emb.astype(np.float32)
                self.documents = docs
                print(f"Loaded {len(self.documents)} documents from storage.")
            else:
                self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
                self.documents = []
                print("No existing vector store found, starting fresh.")
        except Exception as e:
            print(f"Error loading vector store from disk: {e}")
            self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
            self.documents = []

    def _save_to_disk(self):
        """Save embeddings and documents to disk."""
        try:
            if self.embeddings is not None and self.embeddings.shape[0] > 0:
                np.save(self._index_path, self.embeddings.astype(np.float32))
                with open(self._docs_path, "wb") as f:
                    pickle.dump(self.documents, f)
                print(f"Saved {len(self.documents)} documents to storage.")
            else:
                # Remove files if empty
                if self._index_path.exists():
                    try:
                        self._index_path.unlink()
                    except Exception:
                        pass
                if self._docs_path.exists():
                    try:
                        self._docs_path.unlink()
                    except Exception:
                        pass
                print("No documents to save, cleared storage files.")
        except Exception as e:
            print(f"Error saving vector store to disk: {e}")

    def add_documents(self, documents, metadata=None, ids=None):
        """
        Add a list of text documents to the store.
        documents: list[str] or list[dict with 'page_content']
        """
        if not documents:
            print("No documents provided to add.")
            return

        # Normalize documents to dicts with 'page_content'
        normalized_docs = []
        texts = []
        for d in documents:
            if isinstance(d, str):
                normalized_docs.append({"page_content": d})
                texts.append(d)
            elif isinstance(d, dict) and "page_content" in d:
                normalized_docs.append(d)
                texts.append(d["page_content"])
            else:
                # fallback: convert to string
                s = str(d)
                normalized_docs.append({"page_content": s})
                texts.append(s)

        print(f"Adding {len(normalized_docs)} documents to vector store...")

        # encode to numpy (N x dim)
        new_embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        new_embeddings = np.atleast_2d(new_embeddings).astype(np.float32)

        # append embeddings and docs
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(normalized_docs)
        self._save_to_disk()
        print(f"Total documents in store: {len(self.documents)}")

    def _compute_similarities(self, query_embedding):
        """Compute cosine similarities between query_embedding (1xd) and all doc embeddings"""
        if self.embeddings.shape[0] == 0:
            return np.array([])

        # normalize
        q = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        docs_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        sims = (q @ docs_norm.T)[0]  # returns 1D array
        return sims

    def search(self, query, top_k=3):
        """Return top_k similar docs and corresponding similarity scores (as dict)."""
        if len(self.documents) == 0:
            return {"documents": [], "scores": []}

        # generate query embedding
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        sims = self._compute_similarities(q_emb)

        if sims.size == 0:
            return {"documents": [], "scores": []}

        top_indices = np.argsort(sims)[::-1][:top_k]
        docs = [self.documents[i] for i in top_indices]
        scores = [float(sims[i]) for i in top_indices]
        return {"documents": docs, "scores": scores}

    # compatibility helper expected by other code
    def similarity_search_with_score(self, query, k=3):
        """Return list of (doc, score) tuples in decreasing similarity order."""
        res = self.search(query, top_k=k)
        docs = res.get("documents", [])
        scores = res.get("scores", [])
        return list(zip(docs, scores))

    def similarity_search(self, query, k=3):
        """Return only docs (no scores)"""
        res = self.search(query, top_k=k)
        return res.get("documents", [])

    def get_count(self):
        """Number of documents"""
        return len(self.documents)

    def clear(self):
        """Clear in-memory and on-disk store"""
        self.documents = []
        self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        try:
            for file in self._storage_dir.glob("*"):
                try:
                    file.unlink()
                except Exception:
                    pass
            print("Vector store cleared on disk.")
        except Exception as e:
            print(f"Error clearing store: {e}")

# Global singleton instance
_vector_store_instance = None

def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance

def add_to_vector_store(chunks, source_name=None):
    """Add chunks (list of strings or dicts) to store"""
    vs = get_vector_store()
    vs.add_documents(chunks)

# Helper for your QA pipeline to extract text safely
def extract_text_from_doc(d):
    """
    Extract text from a document object or dict.
    Supports dict with 'page_content' or 'text', or plain string.
    """
    try:
        if isinstance(d, str):
            return d
        if isinstance(d, dict):
            return d.get("page_content") or d.get("text") or ""
        # fallback to attribute
        text = getattr(d, "page_content", None)
        if text:
            return text
        text = getattr(d, "text", None)
        if text:
            return text
        return str(d) if d else ""
    except Exception:
        return str(d) if d else ""

# Uncomment below to test loading and adding documents quickly
# if __name__ == "__main__":
#     vs = get_vector_store()
#     print(f"Currently loaded documents: {vs.get_count()}")
#     if vs.get_count() == 0:
#         print("Adding sample documents...")
#         sample_docs = [
#             {"page_content": "This is a test document about AI."},
#             {"page_content": "Another document discussing machine learning."}
#         ]
#         vs.add_documents(sample_docs)
#         print(f"Documents after adding: {vs.get_count()}")
#     else:
#         print("Documents already loaded.")
#     # Test search
#     results = vs.search("machine learning", top_k=2)
#     print("Search results:")
#     for doc, score in zip(results["documents"], results["scores"]):
#         print(f"Score: {score:.3f}, Text: {doc['page_content'][:60]}...")
