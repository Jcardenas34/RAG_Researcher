from flask import Blueprint, request, jsonify
from RAG_researcher.utils import globals
subject_bp = Blueprint('subject', __name__)
# Define the default model and subject


@subject_bp.route('/api/subject', methods=['GET'])
def get_subject():
    """
    Endpoint to get the currently selected LLM.
    """
    return jsonify({"subject": globals.subject_selected})


@subject_bp.route('/api/subject', methods=['POST'])
def set_subject():
    """
    Endpoint to update the selected subject`.
    It expects a JSON payload like: {"subject": "Physics"}
    """
    data = request.get_json()
    
    globals.subject_selected = data.get('subject')
    if not globals.subject_selected:
        return jsonify({"error": "subject name is missing"}), 400
        
    print(f"Selected subject changed to: {globals.subject_selected}")
    return jsonify({"subject": f"subject successfully changed to {globals.subject_selected}"})
