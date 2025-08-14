import os
from flask_cors import CORS
from flask import Flask, request, jsonify
from langchain_postgres import PGVector
from pgvector.psycopg import register_vector
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from datetime import datetime


import logging

'''
Description:
---------------
Runs the backend for a RAG application that will let the user
send a question to an LLM to learn about cutting edge AI research, and


How to run:
---------------
python rag_app.py

open a new terminal and run the webUI  using 
cd sandbox/rag-ui
source start_rag.sh

of manually,

npm run dev


'''

current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y-%m-%d")


# Initialize the logger
custom_logger = logging.getLogger('RAG_logger')
custom_logger.setLevel(logging.INFO)

# Prevent this logger from being passed to the root logger and conaminating file
custom_logger.propagate = False

# Create a file handler for your custom logs
file_handler = logging.FileHandler(f'RAG_{formatted_date}.log')
file_handler.setLevel(logging.INFO)

# Format the output of the logs
formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
file_handler.setFormatter(formatter)

custom_logger.addHandler(file_handler)


# Define the default model and country
llm_model="gpt-4.1"
country_selected="argentina"

# Not importing from local package
class RetrieveApiKey:
    """Class to retrieve API keys for different models."""

    def __init__(self, model:str):
        self.model = model
        print(f"Selected model: {self.model}")
        self.supported_models = {"gpt-4.1":  os.environ.get("OPENAI_API_KEY_PROPETERRA"),
                                 "tinyllama":  os.environ.get("OPENAI_API_KEY_PROPETERRA"),         
                                 "sonar":    os.environ.get("PERPLEXITY_API_KEY_PROPETERRA"),
                                 "sonar-pro":os.environ.get("PERPLEXITY_API_KEY_PROPETERRA"),
                                 "sonar-deep-research":os.environ.get("PERPLEXITY_API_KEY_PROPETERRA"),
                                 "ms_copilot":os.environ.get("MS_COPILOT_API_KEY"),
                                 "claude-sonnet-4-20250514":   os.environ.get("CLAUDE_API_KEY_PROPETERRA"),
                                 "claude-opus-4-20250514": os.environ.get("CLAUDE_API_KEY_PROPETERRA"),
                                 "claude-3-5-haiku-20241022":  os.environ.get("CLAUDE_API_KEY_PROPETERRA"),
                                 "mistral":         os.environ.get("MISTRAL_API_KEY"),
                                 "gemini-2.5-flash":os.environ.get("GEMINI_API_KEY_PROPETERRA"),
                                 "gemini-2.5-pro":  os.environ.get("GEMINI_API_KEY_PROPETERRA"),
                                 "manus":           os.environ.get("MANUS_API_KEY")}

        if model not in self.supported_models:

            raise ValueError("Selected model not in list of supported models.")


# Define state for application (Taken from https://python.langchain.com/docs/tutorials/rag/)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    country_context: str




def initialize(model:str):
    ''' Initializes the model used for querying, as well as the vector database used for storage. '''


    # Initializing necessary parameters, at the global level
    key_retriever = RetrieveApiKey(model)
    # For now, only gpt and gemini models work, given that you have your api keys set in your environment for them
    model_key = key_retriever.supported_models[model]

    # Hardcoded, for gpt, must use this if you want to embed with openai
    openai_key = key_retriever.supported_models["gpt-4.1"]

    if "gpt" in model or "o4-mini" in model:
        llm = init_chat_model(model, model_provider="openai", api_key=model_key)
    elif "gemini" in model:
        llm = init_chat_model(model, model_provider="google_genai", api_key=model_key)
    elif "claude" in model:
        llm = init_chat_model(model, model_provider="anthropic", api_key=model_key)
    elif "tinyllama" in model:
        # Replace with your remote server's IP and the exposed port
        # llm = Ollama(base_url="http://chiralpair@kitsune-cluster.asuscomm.com:11434", model="tinyllama")
        llm = Ollama(base_url="http://localhost:11434", model="tinyllama")
    else:
        print("Please choose a supported model")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_key)

    # Connecting to a postgres database you have defined (must also add teh vector extension! run 'CREATE EXTENSION vector;' once db is open)
    # https://www.postgresql.org/docs/current/tutorial-install.html
    connection = "postgresql+psycopg://chiral@localhost:6543/postgres"
    collection_name = "my_docs"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
    )


    # (Taken from https://python.langchain.com/docs/tutorials/rag/)
    # Define prompt for question-answering
    # N.B. for non-US LangSmith endpoints, you may need to specify
    # api_url="https://api.smith.langchain.com" in hub.pull.
    prompt = hub.pull("rlm/rag-prompt")

    return prompt, llm, vector_store

def index_data(vector_store, site):
    # Scrapes the content of the provided link, must change  class_=("ArticleContent") to relevant sections you want to scrape
    print(f"Indexing site: {site}")
    loader = WebBaseLoader(
        # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        web_paths=((site,)),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                # class_=("ArticleContent")
                class_=("entry-content")
                # class_=("post-title", "post-header", "post-content")
            )
        ),
    )

    docs = loader.load()
    assert len(docs) == 1
    docs[0].metadata["country"] = "argentina"

    # What are the contents of the graph?
    # if args.verbose:
    #     print(f"Total characters: {len(docs[0].page_content)}")
    #     print(docs[0].page_content[:1000])


    # Split text into chunks (tokens) so that they can be vectorized 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Start indexing the chunks
    document_ids = vector_store.add_documents(documents=all_splits)

    # if args.verbose:
        # print(document_ids[:3])



def get_rag_response(question: str, llm_model:str, country:str) -> str:
    """
    This function takes a user prompt, processes it through the RAG system,
    and returns the generated response.

    Args:
        prompt: The input string from the user.

    Returns:
        The RAG model's response as a string.
    """

    # List of sites that were
    indexed_sites = ["",]
    
    
    
    prompt, llm, vector_store = initialize(llm_model)

    # Define application steps (Taken from https://python.langchain.com/docs/tutorials/rag/)
    def retrieve(state: State):
        ''' 
        First function that will run in the lang-chain,
        applies meta data filtering on the vector store to ensure that the 
        correct information is retrived

        Args:
            state: The State object that has the important parts of the lang-chain.
        Returns:
            A dictionary with the top k relevent documents based on a similarity search, filtered by country selected.
        '''
        country_filter = {"country": country.lower()}
        print("applying filter: ", country_filter)
        # Retrieves question from user, and gives top k rrelevant responses to LLM for question answering, applies meta data filtering for more relevant answers
        retrieved_docs = vector_store.similarity_search(query=state["question"], k=5, filter=country_filter) # filter=None will filter by metadata stored, use for only retrieving specific countries, add to meta data
        return {"context": retrieved_docs}
    



    # (Taken from https://python.langchain.com/docs/tutorials/rag/)
    def generate(state: State):
        '''
        Send a query to the LLM selected by passing it the relevant context (the document chunks retrieved in retrieve()
        Stores the question, answer and context in the state object.

        Args:
            state: The State object that has the important parts of the lang-chain.
        Returns:
            A dictionary containing the 
        '''
        # Collecting the relevant content and sources to ensure that ONLY the relevant country information is being used
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        sources = "\n\n".join(str(index)+": "+str(doc.metadata) for index,  doc in enumerate(state["context"]))

        # Where question is actually asked to the LLM
        messages = prompt.invoke({"question": state["question"], "context": docs_content})

        custom_logger.info(f"Juan: {state['question']}\n")
        response = llm.invoke(messages)
        custom_logger.info(f"{llm_model}: {response.content}\n")
        custom_logger.info(f"Context: \n{docs_content}\n")
        custom_logger.info(f"Sources: \n{sources}\n")


        return {"answer": response.content}


    # site = "https://thelatinvestor.com/blogs/news/colombia-real-estate-market"
    # site = "https://thelatinvestor.com/blogs/news/argentina-real-estate-foreigner"
    # site = "https://thelatinvestor.com/blogs/news/panama-city-property"
    # site = "https://www.adventuresincre.com/exploring-latin-americas-real-estate-markets-argentina/"
    site =""

    if site not in indexed_sites:
        print(f"Site {site} not found in database, will load it and parse for an answer to your question.")
        index_data(vector_store, site)
    else:
        print(f"Site {site} found in vector store, loading it..")




    # Compile application and test (Taken from https://python.langchain.com/docs/tutorials/rag/)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Getting the actual response from the model
    response = graph.invoke({"question": question})
    print(response["answer"])

    print(f"Received prompt: {prompt}")

    return response["answer"]
# ---------------------------------


# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This is crucial to allow your React frontend (running on a different port)
# to communicate with this Flask backend.
CORS(app)


@app.route('/api/llm', methods=['GET'])
def get_llm():
    """
    Endpoint to get the currently selected LLM.
    """
    global llm_model
    return jsonify({"llm": llm_model})

@app.route('/api/llm', methods=['POST'])
def set_llm():
    """
    Endpoint to update the selected LLM.
    It expects a JSON payload like: {"llm": "gpt-4.1"}
    """
    global llm_model
    data = request.get_json()
    
    new_llm = data.get('llm')
    if not new_llm:
        return jsonify({"error": "LLM name is missing"}), 400
        
    llm_model = new_llm
    print(f"Selected LLM changed to: {llm_model}")
    return jsonify({"message": f"LLM successfully changed to {llm_model}"})


@app.route('/api/country', methods=['GET'])
def get_country():
    """
    Endpoint to get the currently selected LLM.
    """
    global country_selected
    return jsonify({"country": country_selected})


@app.route('/api/country', methods=['POST'])
def set_country():
    """
    Endpoint to update the selected country`.
    It expects a JSON payload like: {"country": "argentina"}
    """
    global country_selected
    data = request.get_json()
    
    new_country = data.get('country')
    if not new_country:
        return jsonify({"error": "country name is missing"}), 400
        
    country_selected = new_country
    print(f"Selected country changed to: {country_selected}")
    return jsonify({"country": f"country successfully changed to {country_selected}"})


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    This endpoint receives a prompt from the frontend,
    gets a response from the RAG system, and sends it back.
    """
    try:

        global llm_model
        global country_selected

        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the prompt from the data.
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400

        # Get the response from your RAG function
        response_text = get_rag_response(prompt, llm_model, country_selected)

        # Return the response as JSON
        return jsonify({"response": response_text})

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        # Return a generic error message
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    # Host '0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
