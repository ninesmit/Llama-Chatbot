url_gen_response = 'http://localhost:11434/api/chat'
url_get_question = 'http://127.0.0.1:11434/api/generate'
url_eval_response = 'http://127.0.0.1:11434/api/generate'

LLM_gen_response = "llama3.2:3b"
LLM_get_question = "llama3.2:3b"
LLM_eval_response = "gemma2:2b"

embedding_question_model = 'sentence-transformers/all-MiniLM-L6-v2' 
embedding_context_model = 'sentence-transformers/all-MiniLM-L6-v2'

chunk_size = 1500
chunk_overlap = 300
max_output_token = 200
top_k_context = 5
top_k_question = 10

main_prompt = """
    Based on the following information:

    {context}

    Answer the following question:
    {question}

    By following the instruction below
    1.Don't say 'According to the text'
    2.Only answer based on the given context, don't paraphrase
    3.The answer usually locate after the question (Q:), look for it 
    """

evaluate_prompt = """
    Response 1: {expected_response}
    Response 2: {actual_response}

    Compare both responses, focusing on the overall context, not grammar and specific topic only 
    Then, give the confidence score of how much you are confident with the comparison, ranging from 1-10. 1 means low confidence and 10 means high confidence
    """

get_question_prompt = """
    Given a list of question below:
    {question_list}

    Choose the most similar question to this: {question}
    Only generate the most similar question, nothing else is included.
    """