from flask import Blueprint, request, jsonify
from RAG_researcher.utils import globals
from RAG_researcher.utils.helpers import get_rag_response


chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """
    This endpoint receives a prompt from the frontend,
    gets a response from the RAG system, and sends it back.
    """
    try:

        # global llm_model
        # global subject_selected

        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the prompt from the data.
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is missing"}), 400
        
        # Get the response from your RAG function
        response_text = get_rag_response(prompt)

        # Return the response as JSON
        return jsonify({"response": response_text})

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred chat(): {e}")
        # Return a generic error message
        return jsonify({"error": "An internal server error occurred"}), 500
