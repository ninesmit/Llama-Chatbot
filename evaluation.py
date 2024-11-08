from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import requests
from langchain import PromptTemplate
import re
import json
import textwrap
from config import *

URL_EVAL_RESPONSE = url_eval_response
LLM_EVAL_RESPONSE = LLM_eval_response

URL_GET_QUESTION= url_get_question
LLM_GET_QUESTION = LLM_get_question

EMBED_QUESTION_MODEL = embedding_question_model

EVAL_PROMPT = evaluate_prompt
GET_QUESTION_PROMPT = get_question_prompt

def eval_response(wrapped_response, expected_answer):
    eval_prompt = EVAL_PROMPT

    eval_template = PromptTemplate(
        input_variables=["expected_response", "actual_response"],
        template=eval_prompt
    )

    actual_response = wrapped_response
    expected_response = expected_answer

    final_eval_prompt = eval_template.format(expected_response=expected_response, actual_response=actual_response)

    response = requests.post(
        URL_EVAL_RESPONSE,
        json={"model":LLM_EVAL_RESPONSE, "prompt": final_eval_prompt}
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
    # print("Cross-check response:\n", wrapped_answer)
    
    return wrapped_answer

def eval_question(question_list, question):
    compare_prompt = GET_QUESTION_PROMPT

    question_template = PromptTemplate(
        input_variables=["question_list", "question"],
        template=compare_prompt
    )

    question_list = question_list
    question = question

    final_question_prompt = question_template.format(question_list=question_list, question=question)

    response = requests.post(
        URL_GET_QUESTION,
        json={"model":LLM_GET_QUESTION, "prompt": final_question_prompt}
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
    # print("Most similar question:\n", wrapped_answer)
    return wrapped_answer

def extract_questions_answers(document_text):
    # Use regular expression to match Q: <question> followed by <answer> up to the next Q:
    qa_pairs = re.findall(r'Q:\s*(.*?)\n(.*?)(?=Q:|$)', document_text, re.DOTALL)
    
    # Split into questions and answers
    questions = [qa[0].strip() for qa in qa_pairs]
    answers = [qa[1].strip() for qa in qa_pairs]
    
    return questions, answers

def embed_question(question):
    model = HuggingFaceEmbeddings(model_name=EMBED_QUESTION_MODEL)
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
