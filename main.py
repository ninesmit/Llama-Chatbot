from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import ollama
import requests
from langchain import PromptTemplate
import re
import json
import textwrap

def load_document(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents

def split_chunk(document, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(document)
    return chunked_docs

def get_chunk_id(chunked_docs):
    last_page_id = f"{chunked_docs[0].metadata.get('source')}:{chunked_docs[0].metadata.get('page')}"
    current_chunk_index = 0

    for chunk in chunked_docs:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
            last_page_id = current_page_id
        else:
            current_chunk_index = 0
            last_page_id = current_page_id
        chunk_id = f"{source}:{page}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
    return chunked_docs

def embed_to_vectordb(chunked_docs):
    model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma(
        embedding_function=model,
        persist_directory="FAQdb"  
    )
    vectorstore.add_documents(documents=chunked_docs)
    vectorstore.persist()

    return vectorstore

def query(question, vectorstore, top_k):
    query = question
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return results

def get_prompt(results, query):
    template = """
    Based on the following information:

    {context}

    Answer the following question:
    {question}

    By following the instruction below
    1.Don't say 'According to the text'
    2.Only answer based on the given context, don't paraphrase
    3.The answer usually locate after the question (Q:), look for it 
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    context = "\n\n".join([result[0].page_content for result in results])
    question = query
    final_prompt = prompt_template.format(context=context, question=question)
    return final_prompt

def get_response_from_llm(final_prompt, results):
    response = requests.post(
        'http://127.0.0.1:11434/api/generate',
        json={"model":"llama3.2:3b", "prompt": final_prompt}
    )

    lines = response.text.strip().split("\n")

    complete_response = ""

    for line in lines:
        try:
            data = json.loads(line)
            complete_response += data["response"]
        except json.JSONDecodeError:
            print("Failed to parse line:", line)

    # Wrap the response text to 80 characters per line
    wrapped_response = textwrap.fill(complete_response, width=80)

    # Print the wrapped response
    print("Response from LLM:\n", wrapped_response)

    for i, result in enumerate(results, 1):
        print(f"Relevant chunk {i}: {result[0].metadata['id']}")

    return complete_response

def eval_response(wrapped_response, expected_answer):
    eval_prompt = """
    Response 1: {expected_response}
    Response 2: {actual_response}

    Compare both responses, focusing on the meaning, not the grammar
    If they convey the same information, answer 'True'. Otherwise, please answer 'False' with reason. (must answer 'True' or 'False')
    Then, give the confidence score of how much you are confident with the comparison, ranging from 1-10. 1 means low confidence and 10 means high confidence
    """

    eval_template = PromptTemplate(
        input_variables=["expected_response", "actual_response"],
        template=eval_prompt
    )

    actual_response = wrapped_response
    expected_response = expected_answer

    final_eval_prompt = eval_template.format(expected_response=expected_response, actual_response=actual_response)

    response = requests.post(
        'http://127.0.0.1:11434/api/generate',
        json={"model":"llama3.2:3b", "prompt": final_eval_prompt}
    )

    lines = response.text.strip().split("\n")

    complete_response = ""

    for line in lines:
        try:
            data = json.loads(line)
            complete_response += data["response"]
        except json.JSONDecodeError:
            print("Failed to parse line:", line)

    # Wrap the response text to 80 characters per line
    wrapped_answer = textwrap.fill(complete_response, width=80)

    # Print the wrapped response
    print("Cross-check response:\n", wrapped_answer)
    
    return wrapped_answer

def extract_questions_answers(document_text):
    # Use regular expression to match Q: <question> followed by <answer> up to the next Q:
    qa_pairs = re.findall(r'Q:\s*(.*?)\n(.*?)(?=Q:|$)', document_text, re.DOTALL)
    
    # Split into questions and answers
    questions = [qa[0].strip() for qa in qa_pairs]
    answers = [qa[1].strip() for qa in qa_pairs]
    
    return questions, answers

def embed_question(question):
    model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = [q for q in question]
    vectorstore = Chroma(
        embedding_function=model,
        persist_directory="Qdb"  
    )
    vectorstore.add_texts(documents)
    vectorstore.persist()
    return vectorstore

def get_question(input_question, vectorstore, top_k):
    results = vectorstore.similarity_search_with_score(input_question, k=top_k)
    return results

def eval_question(question_list, question):
    compare_prompt = """
    Given a list of question below:
    {question_list}

    Choose the most similar question to this: {question}
    Only generate the most similar question, nothing else is included.
    """

    question_template = PromptTemplate(
        input_variables=["question_list", "question"],
        template=compare_prompt
    )

    question_list = question_list
    question = question

    final_question_prompt = question_template.format(question_list=question_list, question=question)

    response = requests.post(
        'http://127.0.0.1:11434/api/generate',
        json={"model":"llama3.2:3b", "prompt": final_question_prompt}
    )
    
    lines = response.text.strip().split("\n")

    complete_response = ""

    for line in lines:
        try:
            data = json.loads(line)
            complete_response += data["response"]
        except json.JSONDecodeError:
            print("Failed to parse line:", line)

    # Wrap the response text to 80 characters per line
    wrapped_answer = textwrap.fill(complete_response, width=80)

    # Print the wrapped response
    print("Most similar question:\n", wrapped_answer)
    return wrapped_answer

def classify_intent(user_input):
    # Convert input to lowercase for case-insensitive matching
    user_input = user_input.lower().strip()

    # 1. Greeting Intent: Check for common greetings
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening']
    if user_input in greetings:
        return 'greeting'

    # 2. Asking Question Intent: Check if the input starts with a question word
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'is', 'do', 'does']
    if any(user_input.startswith(word) for word in question_words):
        return 'ask_question'

    # 3. Incomplete Sentence Intent: Check if input is too short (e.g., less than 3 words)
    if len(user_input.split()) < 2:
        return 'incomplete_sentence'

    # 4. Out of Scope Intent: If no rules match, classify as "out_of_scope"
    return 'out_of_scope'


# LLaMA API URL (make sure the server is running)
LLAMA_API_URL = "http://localhost:11434/api/chat"

def llama_chat():

    # First Phase
    documents = load_document('AmazonFAQ.pdf')
    chunked_docs = split_chunk(documents, 1500, 300)
    chunked_docs = get_chunk_id(chunked_docs) 
    vectorstore = embed_to_vectordb(chunked_docs)

    combined_text = "\n".join([doc.page_content for doc in documents])
    questions, answers = extract_questions_answers(combined_text)

    question_db = embed_question(questions)

    # Second Phase
    conversation_history = []
    print("LLaMA Chatbot: Start your conversation! Type 'exit' to quit.\n")

    while True:
        # Take user input
        user_input = input("You: ")
        input_class = classify_intent(user_input)
        print('This is class:', input_class)

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if input_class == 'greeting':
            # Add the user's input to the conversation history
            conversation_history.append({"role": "user", "content": user_input})
        elif input_class == 'ask_question':

            search_result = query(user_input, vectorstore, top_k=3)
            final_prompt = get_prompt(search_result, user_input)
            
            # Debugging
            print('This is the final prompt:', final_prompt)
            for i, result in enumerate(search_result, 1):
                print(f"Relevant chunk {i}: {result[0].metadata['id']}")
            
            # Add the user's input to the conversation history
            conversation_history.append({"role": "user", "content": final_prompt})

        elif input_class == 'incomplete_sentence':
            # Add the user's input to the conversation history
            conversation_history.append({"role": "user", "content": user_input})
        else:
            # Add the user's input to the conversation history
            conversation_history.append({"role": "user", "content": user_input})

        # Prepare the data to send to the /chat endpoint
        payload = {
            "model": "llama3.2:3b",  # Specify the model
            "messages": conversation_history,
            "max_tokens": 200  # Adjust based on your need
        }

        try:
            # Send a POST request to the LLaMA API with streaming enabled
            with requests.post(
                LLAMA_API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True  # Enable streaming response
            ) as response:
                # Check if the response was successful
                if response.status_code != 200:
                    print(f"Error: {response.status_code}, {response.text}")
                    continue

                # Initialize a variable to store the complete assistant's reply
                assistant_reply = ""

                # Process each line of the streaming response
                for line in response.iter_lines():
                    if line:
                        # Parse the JSON object from each line
                        message = json.loads(line)
                        # Extract the assistant's content and accumulate the reply
                        assistant_reply += message.get("message", {}).get("content", "")

                # Print the full assistant reply
                print(f"LLaMA: {assistant_reply}\n")

                # Add the assistant's reply to the conversation history
                conversation_history.append({"role": "assistant", "content": assistant_reply})

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print("The response was not valid JSON. Check the raw response.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    llama_chat()
