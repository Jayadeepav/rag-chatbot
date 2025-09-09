from vector_store import get_vector_store, add_to_vector_store

def test_add_and_search():
    vs = get_vector_store()
    print(f"Initially loaded documents: {vs.get_count()}")

    if vs.get_count() == 0:
        print("Adding sample documents...")
        sample_docs = [
            {"page_content": "This is a test document about AI."},
            {"page_content": "Another document discussing machine learning."}
        ]
        add_to_vector_store(sample_docs)
        print(f"Documents after adding: {vs.get_count()}")
    else:
        print("Documents already loaded.")

    # Test search
    results = vs.search("machine learning", top_k=2)
    print("Search results:")
    for doc, score in zip(results["documents"], results["scores"]):
        print(f"Score: {score:.3f}, Text: {doc['page_content'][:60]}...")

if __name__ == "__main__":
    test_add_and_search()