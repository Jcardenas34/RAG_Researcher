import sys
from flask import Flask
from flask_cors import CORS
from RAG_researcher.utils import globals
from RAG_researcher.utils.helpers import initialize
from RAG_researcher.utils.logger import rag_logger
from RAG_researcher.routes.chat import chat_bp
from RAG_researcher.routes.configure_llm import llm_bp
from RAG_researcher.routes.configure_subject import subject_bp

'''

Description:
---------------
Runs the backend for a RAG application that will let the user
send a question to an LLM to learn about cutting edge AI research, and math, chemistry, physics topics.


How to run:
---------------
RAG-researcher

open a new terminal and run the webUI  using 
cd /src/RAG_researcher/rag-ui
source start_rag.sh

of manually,

cd /src/RAG_researcher/rag-ui
npm run dev


'''



globals.prompt, globals.llm, globals.vector_store = initialize(globals.llm_model)

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This is crucial to allow your React frontend (running on a different port)
# to communicate with this Flask backend.
CORS(app)

app.register_blueprint(chat_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(subject_bp)


def main() -> int:
    """
    Main function to run the RAG application.
    It initializes the vector store and loads documents.
    """
    # Host '0.0.0.0' makes it accessible on your local network
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
        return 1
    except KeyboardInterrupt:
        print("Server stopped by user.")
        return 0
    except Exception as e:
        rag_logger.error(f"Error starting the Flask app: {e}")
        print(f"Error starting the Flask app: {e}")
        return 1

    
if __name__ == '__main__':
    
    sys.exit(main())