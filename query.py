def query(question, vectorstore, top_k):
    query = question
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return results