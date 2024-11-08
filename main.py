import requests
import json
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from preprocess import load_document, split_chunk, get_chunk_id, embed_to_vectordb
from query import query
from prompt import get_prompt, classify_intent
from evaluation import embed_question, get_question, extract_questions_answers, eval_question, eval_response
from config import *

LLM_API_URL = url_gen_response
LLM_GEN_RESPONSE = LLM_gen_response

EMBED_CONTEXT_MODEL = embedding_context_model
EMBED_QUESTION_MODEL = embedding_question_model

def llama_chat():

    # First Phase
    documents = load_document('AmazonFAQ.pdf')
    combined_text = "\n".join([doc.page_content for doc in documents])
    questions, answers = extract_questions_answers(combined_text)

    if os.path.isdir('FAQdb'):
        model = HuggingFaceEmbeddings(model_name=EMBED_CONTEXT_MODEL)
        context_db = Chroma(
            embedding_function=model,
            persist_directory="FAQdb"  
        )
    else:
        chunked_docs = split_chunk(documents, chunk_size, chunk_overlap)
        chunked_docs = get_chunk_id(chunked_docs) 
        context_db = embed_to_vectordb(chunked_docs)

    if os.path.isdir('Qdb'):
        model = HuggingFaceEmbeddings(model_name=EMBED_QUESTION_MODEL)
        question_db = Chroma(
            embedding_function=model,
            persist_directory="Qdb"  
        )
    else:
        question_db = embed_question(questions)

    # Second Phase
    conversation_history = []
    print("LLaMA Chatbot: Start your conversation! Type 'exit' to quit.\n")

    while True:
        # Take user input
        user_input = input("You: ")
        input_class = classify_intent(user_input)
        print('This user input is in class:', input_class)

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if input_class == 'greeting':
            # Add the user's input to the conversation history
            conversation_history.append({"role": "user", "content": user_input})
            compare_status = False
        elif input_class == 'ask_question':

            search_result = query(user_input, context_db, top_k=top_k_context)
            final_prompt = get_prompt(search_result, user_input)
            
            # Debugging
            print('Step 1: Retrieving the relevant chunks of text ...')
            for i, result in enumerate(search_result, 1):
                print(f"Relevant chunk {i}: {result[0].metadata['id']}")
            print('-------------------------------------------------')

            print('Step 2: Querying the most relavant question from the database')
            question_result = get_question(user_input, question_db, top_k=top_k_question)
            question_list = [q[0].page_content for q in question_result]

            final_question = question_list[0]
            # final_question = eval_question(question_list, user_input)

            final_question = final_question.replace("'", "")
            final_question = final_question.replace(".", "")
            final_question = final_question.replace('"', "")
            print('The most similar question:', final_question)
            print('-------------------------------------------------')

            try:
                question_index = questions.index(final_question)
                answer = answers[question_index]
                compare_status = True
            except ValueError:
                print('Cannot find the relevant question to compare')
                compare_status = False
            
            # Add the user's input to the conversation history
            conversation_history.append({"role": "user", "content": final_prompt})

        elif input_class == 'incomplete_sentence':
            conversation_history.append({"role": "user", "content": user_input})
            compare_status = False
        else:
            conversation_history.append({"role": "user", "content": user_input})
            compare_status = False

        payload = {
            "model": LLM_GEN_RESPONSE,
            "messages": conversation_history,
            "max_tokens": max_output_token 
        }

        try:
            print('Step 3: Getting response from the LLM ...')
            with requests.post(
                LLM_API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True
            ) as response:
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
                print('-------------------------------------------------')

                # Add the assistant's reply to the conversation history
                conversation_history.append({"role": "assistant", "content": assistant_reply})

                if compare_status == True:
                    final_response = conversation_history[-1]['content']
                    answer_check = eval_response(final_response, answer)
                    print('Step 4: Comparing with the ground truth using another LLM')
                    print('Result from the comparison:', answer_check)
                    print('-------------------------------------------------')

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print("The response was not valid JSON. Check the raw response.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    llama_chat()
