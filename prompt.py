from langchain import PromptTemplate
from config import *

MAIN_PROMPT = main_prompt

def get_prompt(results, query):
    template = MAIN_PROMPT

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    context = "\n\n".join([result[0].page_content for result in results])
    question = query
    final_prompt = prompt_template.format(context=context, question=question)
    return final_prompt

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