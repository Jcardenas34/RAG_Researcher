from flask import Blueprint, request, jsonify
from RAG_researcher.utils import globals

llm_bp = Blueprint('llm', __name__)


@llm_bp.route('/api/llm', methods=['GET'])
def get_llm():
    """
    Endpoint to get the currently selected LLM.
    """

    return jsonify({"llm": globals.llm_model})

@llm_bp.route('/api/llm', methods=['POST'])
def set_llm():
    """
    Endpoint to update the selected LLM.
    It expects a JSON payload like: {"llm": "gpt-4.1"}
    """

    data = request.get_json()
    
    globals.llm_model = data.get('llm')
    if not globals.llm_model:
        return jsonify({"error": "LLM name is missing"}), 400
        

    print(f"Selected LLM changed to: {globals.llm_model}")
    return jsonify({"message": f"LLM successfully changed to {globals.llm_model}"})
