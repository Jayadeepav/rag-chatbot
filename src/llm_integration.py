import os
from typing import Optional

class LLMClient:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_local = model_path and os.path.exists(model_path)
        self._vector_store = None  # Add vector store reference
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using enhanced prompt engineering"""
        return self._generate_with_advanced_prompt(query, context)
    
    def _generate_with_advanced_prompt(self, query: str, context: str) -> str:
        """If no context found, search through all documents"""
        if not context.strip():
            # Emergency fallback - search through all documents
            if self._vector_store and hasattr(self._vector_store, 'documents'):
                all_docs = self._vector_store.documents
                if all_docs and len(all_docs) > 0:
                    print("⚠️ Using emergency fallback - no context found from search")
                    full_context = "\n\n".join(all_docs[:3])  # First 3 docs
                    # Create a basic prompt for the fallback
                    prompt = f"Context: {full_context}\nQuestion: {query}\nAnswer:"
                    return self._generate_intelligent_response(query, full_context, prompt)
        
        # Normal processing with the actual context
        prompt = f"""**INSTRUCTION**: Answer the question based ONLY on the provided context. Follow these rules strictly:

1. **ANSWER ONLY FROM CONTEXT**: Use only information present in the context below
2. **BE PRECISE**: Provide specific, factual answers
3. **NO FABRICATION**: If the answer isn't in the context, say "I cannot find this information in the documents"
4. **BE CONCISE**: Keep answers brief and to the point
5. **QUOTE WHEN POSSIBLE**: Use exact phrases from the context when appropriate

**CONTEXT**:
{context}

**QUESTION**:
{query}

**ANSWER**:"""
        
        return self._generate_intelligent_response(query, context, prompt)
    
    def _generate_intelligent_response(self, query: str, context: str, prompt: str) -> str:
        """Improved response generation with better context matching"""
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Check if context is relevant to the question
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())
        relevant_words = query_words.intersection(context_words)
        
        if not relevant_words:
            return "I cannot find relevant information in the documents to answer this question."
        
        # Extract sentences that contain query words
        sentences = context.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            # Check if sentence contains any query words
            contains_query_words = any(word in sentence_lower for word in query_words)
            # Check if sentence is relevant to the query
            is_relevant = (contains_query_words and 
                          len(sentence) > 10 and  # Avoid very short sentences
                          not sentence_lower.startswith(('http', 'www.')))  # Avoid URLs
            
            if is_relevant:
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            return "I found some related information, but nothing that directly answers your question."
        
        # Prioritize sentences with more query words
        def relevance_score(sentence):
            sentence_lower = sentence.lower()
            return sum(1 for word in query_words if word in sentence_lower)
        
        relevant_sentences.sort(key=relevance_score, reverse=True)
        
        # Combine the most relevant sentences
        if len(relevant_sentences) == 1:
            return relevant_sentences[0] + "."
        else:
            return ". ".join(relevant_sentences[:3]) + "."
    
    def _generate_with_phi2(self, query: str, context: str, prompt: str) -> str:
        """Placeholder for when you get Phi-2 working"""
        # This will be used when you install llama-cpp successfully
        try:
            # Your Phi-2 integration code will go here
            return self._generate_intelligent_response(query, context, prompt)
        except:
            return self._generate_intelligent_response(query, context, prompt)

def validate_response_quality(answer: str, query: str, context: str) -> dict:
    """Validate the quality of the generated response"""
    quality_metrics = {
        "relevance": 0.0,
        "specificity": 0.0,
        "completeness": 0.0,
        "confidence": 0.0
    }
    
    # Simple quality checks
    answer_lower = answer.lower()
    query_lower = query.lower()
    context_lower = context.lower()
    
    # Relevance: Check if answer contains query words
    query_words = set(query_lower.split())
    answer_words = set(answer_lower.split())
    common_words = query_words.intersection(answer_words)
    quality_metrics["relevance"] = len(common_words) / len(query_words) if query_words else 0
    
    # Specificity: Check if answer is specific (not generic)
    generic_phrases = ["i don't know", "not sure", "cannot find", "no information"]
    is_generic = any(phrase in answer_lower for phrase in generic_phrases)
    quality_metrics["specificity"] = 0.0 if is_generic else 0.7  # Base score
    
    # Add points for containing numbers, specific terms
    if any(char.isdigit() for char in answer):
        quality_metrics["specificity"] += 0.2
    if len(answer.split()) > 5:  # Not too short
        quality_metrics["specificity"] += 0.1
    
    # Cap at 1.0
    quality_metrics["specificity"] = min(quality_metrics["specificity"], 1.0)
    
    return quality_metrics

# Global instance
_llm_client = None

def get_llm_client():
    """Get or create LLM client instance"""
    global _llm_client
    if _llm_client is None:
        model_path = "models/phi-2-GGUF/phi-2.Q4_K_M.gguf"
        _llm_client = LLMClient(model_path)
        
        # Add vector store reference to LLM client for emergency fallback
        from src.vector_store import get_vector_store
        _llm_client._vector_store = get_vector_store()
    
    return _llm_client

def generate_response(query: str, context: str) -> str:
    """Generate response using available LLM with quality validation"""
    client = get_llm_client()
    answer = client.generate_response(query, context)
    
    # Validate response quality
    quality = validate_response_quality(answer, query, context)
    
    # Add quality indicator for very low quality responses
    if quality["relevance"] < 0.2 or quality["specificity"] < 0.3:
        answer = f"{answer}\n\n*⚠️ Low confidence: This answer may not be fully relevant to your question.*"
    
    return answer