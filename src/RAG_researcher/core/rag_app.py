import sys
from flask_cors import CORS
from flask import Flask, request, jsonify



from datetime import datetime
from RAG_researcher.utils.helpers import RetrieveApiKey
from RAG_researcher.utils.helpers import State
from RAG_researcher.utils.helpers import get_rag_response


import logging

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


# Define the default model and subject
llm_model="gpt-4.1"
subject_selected="ML"



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


@app.route('/api/subject', methods=['GET'])
def get_subject():
    """
    Endpoint to get the currently selected LLM.
    """
    global subject_selected
    return jsonify({"subject": subject_selected})


@app.route('/api/subject', methods=['POST'])
def set_subject():
    """
    Endpoint to update the selected subject`.
    It expects a JSON payload like: {"subject": "Physics"}
    """
    global subject_selected
    data = request.get_json()
    
    new_subject = data.get('subject')
    if not new_subject:
        return jsonify({"error": "subject name is missing"}), 400
        
    subject_selected = new_subject
    print(f"Selected subject changed to: {subject_selected}")
    return jsonify({"subject": f"subject successfully changed to {subject_selected}"})


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    This endpoint receives a prompt from the frontend,
    gets a response from the RAG system, and sends it back.
    """
    try:

        global llm_model
        global subject_selected

        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the prompt from the data.
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400

        # Get the response from your RAG function
        response_text = get_rag_response(prompt, llm_model, subject_selected)

        # Return the response as JSON
        return jsonify({"response": response_text})

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        # Return a generic error message
        return jsonify({"error": "An internal server error occurred"}), 500

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
        custom_logger.error(f"Error starting the Flask app: {e}")
        print(f"Error starting the Flask app: {e}")
        return 1

    
if __name__ == '__main__':
    
    sys.exit(main())