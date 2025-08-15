import os
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from RAG_researcher.utils import globals
from RAG_researcher.utils.logger import rag_logger

from langchain_postgres import PGVector
from pgvector.psycopg import register_vector
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from typing_extensions import List, TypedDict


class RetrieveApiKey:
    """Class to retrieve API keys for different models."""

    def __init__(self, model:str):
        self.model = model
        print(f"Selected model: {self.model}")
        self.supported_models = {"gpt-4.1":  os.environ.get("OPENAI_API_KEY"),
                                 "gpt-5":  os.environ.get("OPENAI_API_KEY"),
                                 "tinyllama":  os.environ.get("OPENAI_API_KEY"),
                                 "gemma3:270m": os.environ.get("OPENAI_API_KEY"),         
                                 "sonar":    os.environ.get("PERPLEXITY_API_KEY"),
                                 "sonar-pro":os.environ.get("PERPLEXITY_API_KEY"),
                                 "sonar-deep-research":os.environ.get("PERPLEXITY_API_KEY"),
                                 "ms_copilot":os.environ.get("MS_COPILOT_API_KEY"),
                                 "claude-sonnet-4-20250514":   os.environ.get("CLAUDE_API_KEY"),
                                 "claude-opus-4-20250514": os.environ.get("CLAUDE_API_KEY"),
                                 "claude-3-5-haiku-20241022":  os.environ.get("CLAUDE_API_KEY"),
                                 "mistral":         os.environ.get("MISTRAL_API_KEY"),
                                 "gemini-2.5-flash":os.environ.get("GEMINI_API_KEY"),
                                 "gemini-2.5-pro":  os.environ.get("GEMINI_API_KEY"),
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
        # Allows dor querying the tinyllama model, you are hosting locally
        llm = Ollama(base_url="http://localhost:11434", model="tinyllama")
    elif "gemma3:270m" in model:
        # Allows dor querying the tinyllama model, you are hosting locally
        llm = Ollama(base_url="http://localhost:11434", model="gemma3:270m")
    else:
        print("Please choose a supported model")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_key)

    # Connecting to a postgres database you have defined (must also add the vector extension! run 'CREATE EXTENSION vector;' once db is open)
    # helpful link: https://www.postgresql.org/docs/current/tutorial-install.html
    connection = "postgresql+psycopg://chiral@localhost:6543/postgres"
    collection_name = "my_docs"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
    )


    # (Taken from https://python.langchain.com/docs/tutorials/rag/) defult RAG prompt
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
    docs[0].metadata["subject"] = globals.subject_selected.lower()
    # docs[0].metadata["subject"] = "Physics"

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

def retrieve(state: State):
    ''' 
    First function that will run in the lang-chain,
    applies meta data filtering on the vector store to ensure that the 
    correct information is retrived

    Args:
        state: The State object that has the important parts of the lang-chain.
    Returns:
        A dictionary with the top k relevent documents based on a similarity search, filtered by subject selected.
    '''
    subject = {"subject": globals.subject_selected.lower()}
    print("applying filter: ", subject)
    # Retrieves question from user, and gives top k rrelevant responses to LLM for question answering, applies meta data filtering for more relevant answers
    retrieved_docs = globals.vector_store.similarity_search(query=state["question"], k=5, filter=subject) # filter=None will filter by metadata stored, use for only retrieving specific countries, add to meta data
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
    # Collecting the relevant content and sources to ensure that ONLY the relevant subject information is being used
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    sources = "\n\n".join(str(index)+": "+str(doc.metadata) for index,  doc in enumerate(state["context"]))

    # Where question is actually asked to the LLM
    messages = globals.prompt.invoke({"question": state["question"], "context": docs_content})

    rag_logger.info(f"Juan: {state['question']}\n")
    response = globals.llm.invoke(messages)

    if globals.llm_model == "tinyllama" or globals.llm_model == "gemma3:270m":
        content = response
    else:
        content = response.content

    rag_logger.info(f"{globals.llm_model}: {content}\n")
    rag_logger.info(f"Context: \n{docs_content}\n")
    rag_logger.info(f"Sources: \n{sources}\n")


    return {"answer": content}


def get_rag_response(question: str) -> str:
    """
    This function takes a user prompt, processes it through the RAG system,
    and returns the generated response.

    Args:
        prompt: The input string from the user.

    Returns:
        The RAG model's response as a string.
    """

    # List of sites that were
    indexed_sites = [""]
    
    
    # Define application steps

    site =""

    if site not in indexed_sites:
        print(f"Site {site} not found in database, will load it and parse for an answer to your question.")
        index_data(globals.vector_store, site)
    else:
        print(f"Site {site} found in vector store, loading it..")




    # Compile application and test (Taken from https://python.langchain.com/docs/tutorials/rag/)
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Getting the actual response from the model
    response = graph.invoke({"question": question})
    print(response["answer"])

    print(f"Received prompt: {question}")

    return response["answer"]
# ---------------------------------
