from transformers import pipeline
from src.vector_store import get_vector_store

# Helper: safe text extraction
def _extract_text_from_doc(d):
    try:
        text = getattr(d, "page_content", None)
        if not text:
            text = d.get("page_content", None) if isinstance(d, dict) else None
        if not text:
            text = d.get("text", None) if isinstance(d, dict) else None
        return text if text else ""
    except Exception:
        return str(d) if d else ""

# Retrieve documents from your vector store
def retrieve_docs(query, top_k=3):
    vs = get_vector_store()
    results = vs.similarity_search_with_score(query, k=top_k)

    docs = [doc for doc, score in results]
    similarities = [float(score) for doc, score in results]

    return docs, similarities

# Main query function
def query_documents(query, top_k=3):
    """
    Retrieve documents, extract context, and answer the query.
    """
    # 1) Retrieve from DB
    docs, similarities = retrieve_docs(query, top_k=top_k)

    # 2) Extract clean text
    contexts = [_extract_text_from_doc(d) for d in docs]

    # 3) Guard: no contexts
    if not contexts or all(c.strip() == "" for c in contexts):
        return {
            "answer": "No documents found in the knowledge base. Please add documents first.",
            "contexts": contexts,
            "similarities": similarities,
            "confidence": None
        }

    # 4) QA with HuggingFace pipeline
    try:
        qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

        combined_for_qa = " ".join([c for c in contexts if c and len(c.strip()) > 20])
        if not combined_for_qa.strip():
            return {
                "answer": "No valid text retrieved from the documents.",
                "contexts": contexts,
                "similarities": similarities,
                "confidence": None
            }

        res = qa(question=query, context=combined_for_qa)
        answer = res.get("answer", "").strip()
        confidence = float(res.get("score", 0.0))

    except Exception as e:
        answer = f"Error: QA pipeline failed ({e})."
        confidence = None

    # ✅ Final return
    return {
        "answer": answer,
        "contexts": contexts,
        "similarities": similarities,
        "confidence": confidence
    }
