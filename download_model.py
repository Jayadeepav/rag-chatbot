# download_model.py
# Run from project root: python download_model.py
from sentence_transformers import SentenceTransformer
import os

def download_embeddings_model():
    target = "models/all-MiniLM-L6-v2"
    if os.path.isdir(target):
        print(f"Embeddings model already exists at {target}")
        return
    print("Downloading sentence-transformers/all-MiniLM-L6-v2 ... (this may take a while)")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    os.makedirs(target, exist_ok=True)
    model.save(target)
    print(f"Saved embeddings model to {target}")

if __name__ == "__main__":
    download_embeddings_model()
