from src.vector_store import get_vector_store
from src.llm_integration import generate_response

def query_documents(query, top_k=5):  # Increased top_k for better results
    """Query documents using RAG approach with confidence scoring"""
    # Get relevant documents from vector store
    vector_store = get_vector_store()
    results = vector_store.search(query, top_k=top_k)
    
    # Extract contexts and similarities - ensure they are lists
    contexts = results.get('documents', [[]])[0]
    similarities = results.get('distances', [[]])[0]
    
    # Ensure similarities is always a list
    if not isinstance(similarities, list):
        similarities = [similarities] if similarities is not None else []
    
    # Calculate average similarity score (confidence)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # Generate response
    if contexts:
        combined_context = "\n\n".join(contexts)
        answer = generate_response(query, combined_context)
        
        # Add confidence metadata to answer
        if avg_similarity < 0.4:  # Low confidence
            answer = f"{answer}\n\n*Note: I'm not very confident about this answer as the context isn't strongly related.*"
        elif avg_similarity > 0.7:  # High confidence
            answer = f"{answer}\n\n*This information appears to be well-supported by the documents.*"
            
    else:
        answer = "No relevant documents found. Please add documents to the knowledge base first."
        contexts = []
        similarities = []
    
    return {
        "answer": answer,
        "contexts": contexts,
        "confidence": avg_similarity,
        "similarities": similarities  # Ensure this is always a list
    }