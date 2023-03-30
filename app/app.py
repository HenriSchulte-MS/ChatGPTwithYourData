from flask import Flask, render_template, request
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import pandas as pd
import numpy as np
import os
from pathlib import Path
from transformers import GPT2TokenizerFast


# Chat Markup Language tokens
SYSTEM_MSG_TOKEN = '<|im_start|>system'
ASSISTANT_MSG_TOKEN = '<|im_start|>assistant'
USER_MSG_TOKEN = '<|im_start|>user'
END_MSG_TOKEN = '<|im_end|>'

# Model deployment names
DOC_EMBEDDINGS_MODEL = 'text-search-curie-doc-001'
QUERY_EMBEDDINGS_MODEL = 'text-search-curie-query-001'
CHAT_MODEL = 'gpt-35-turbo'

# Number of knowledge base articles to include in prompt
CONTEXT_SIZE = 1

# Key vault secrets names
OPENAI_KEY_NAME = 'AzureOAIKeyUS'
OPENAI_ENDPOINT_NAME = 'AzureOAIEndpointUS'

# Path to knowledge base
KB_PATH = 'static/json/kb.json'

# Prompt 
COMPANY_NAME = 'Contoso'
COMPANY_LOCATION = 'Denmark'
COMPANY_DESCRIPTION = 'online retailer in Denmark'
TOPICS = 'products and usage instructions'


def get_keyvault_secret(secret_name):
    keyvault_uri = os.getenv('KEYVAULT_URI')
    if keyvault_uri is None:
        raise ValueError('KEYVAULT_URI environment variable not set.')
    credential = DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True,
            exclude_shared_token_cache_credential=True)
    keyvault_client = SecretClient(vault_url=keyvault_uri, credential=credential)
    secret = keyvault_client.get_secret(secret_name)
    return secret


def construct_prompt(query: str, kb: pd.DataFrame):
    """
    Construct a prompt for the OpenAI API based on the query and the knowledge base.
    """
    # Get the most relevant knowledge base articles
    kb = order_knowledge_base_by_query_similarity(query, kb)

    # Concatenate knowledge base articles without line breaks up to the context size
    kb['content'] = kb['content'].str.replace('\n', '')
    kb_articles = '\n'.join(kb['content'].head(CONTEXT_SIZE).tolist())

    prompt = (
            f'{SYSTEM_MSG_TOKEN}\nAssistant is a polite, helpful chatbot for {COMPANY_NAME}, a {COMPANY_DESCRIPTION}. '
            f'Assistant can help customers with questions about {TOPICS}. '
            'User is located in Denmark, but assistant replies in the language they use.'
            'Assistant answers questions only based on information from the knowledge base. '
            'Else, assistant says "I don\'t know". Assistant refuses to discuss topics not covered in the knowledge base.'
            f'\nKnowledge base:\n{kb_articles}\n{END_MSG_TOKEN}\n'
        )

    # Add message history to prompt
    for i, m in enumerate(messages):
        token = ASSISTANT_MSG_TOKEN if i % 2 == 0 else USER_MSG_TOKEN
        prompt += f'{token}\n{m}\n{END_MSG_TOKEN}\n'

    return prompt + '\n' + ASSISTANT_MSG_TOKEN


def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      engine=model,
      input=text
    )
    return result["data"][0]["embedding"]


def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))


def order_knowledge_base_by_query_similarity(query: str, kb: pd.DataFrame) -> pd.DataFrame:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    # Sort df by similarity in descending order
    kb['similarity'] = kb['embeddings'].apply(lambda x: vector_similarity(query_embedding, x))
    kb = kb.sort_values(by='similarity', ascending=False)
    
    return kb


def generate_knowledge_base_embeddings():
    # Create empty knowledge base
    kb = {
        'title': [],
        'heading': [],
        'content': []
    }

    # Read in all files in the raw data folder
    for child in Path('data/').iterdir():
        if child.is_file():
            #print(f"{child.name}:\n{child.read_text(encoding='utf-8')}\n")
            split_title = child.name.split(';')
            kb['title'].append(split_title[0])
            kb['heading'].append(split_title[1] if len(split_title) > 1 else '')
            kb['content'].append(child.read_text(encoding='utf-8'))

    # Transform knowledge base into a Pandas dataframe
    df = pd.DataFrame(kb)
    df = df.set_index(['title', 'heading'])

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Count the number of tokens in each document
    df['tokens'] = df.apply(lambda x: count_tokens(x.content, tokenizer), axis=1)

    # Compute document embeddings
    df['embeddings'] = compute_doc_embeddings(df)

    # Write dataframe to JSON file
    df.to_json(KB_PATH, orient='records')


def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return [get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()]


def count_tokens(text: str, tokenizer) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))


# Get OpenAI API key from Key Vault
openai_key = get_keyvault_secret(OPENAI_KEY_NAME).value
openai_endpoint = get_keyvault_secret(OPENAI_ENDPOINT_NAME).value

# Set OpenAI parameters
openai.api_key = openai_key
openai.api_base = openai_endpoint
openai.api_type = 'azure'
openai.api_version = '2022-12-01' # this may change in the future

# Read in knowledge base
if not os.path.exists(KB_PATH):
    # Generate knowledge base embeddings if not already generated
    print('Generating knowledge base embeddings...')
    generate_knowledge_base_embeddings()
kb = pd.read_json(KB_PATH, orient='records')

# Create Flask app
app = Flask(__name__)

# Set up initial message
start_msg = f'Hi, I\'m the {COMPANY_NAME} assistant. Can I help you with anything?'
messages = [start_msg]


@app.route('/')
def hello_world():
    return render_template('index.html', messages=messages)


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Get chat message from user
    message = request.form['message']
    messages.append(message)

    # Construct prompt
    prompt = construct_prompt(message, kb)
    print(prompt)
    
    # Get response from OpenAI
    response = openai.Completion.create(
        engine=CHAT_MODEL,
        prompt=prompt,
        max_tokens=256,
        temperature=0.9
    )
    # Get the first response from the list of choices
    assistant_reply = response['choices'][0]['text']

    # Remove the end message token from the response
    assistant_reply = assistant_reply.replace(END_MSG_TOKEN, '')

    # Add the response to the list of messages
    messages.append(assistant_reply)

    return render_template('index.html', messages=messages)


@app.route('/reset', methods=['GET', 'POST'])
def reset():
    messages.clear()
    messages.append(start_msg)
    return render_template('index.html', messages=messages)