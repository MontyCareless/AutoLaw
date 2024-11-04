from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the Hugging Face API key from environment variables
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Endpoint to analyze legal text
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()  # Get the JSON data from the client (React frontend)
    text = data.get("text")    # Extract the "text" field

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Call Hugging Face API with the LegalBERT model
        response = requests.post(
            "https://api-inference.huggingface.co/models/nlpaueb/legal-bert-base-uncased",
            headers={"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"},
            json={"inputs": text}
        )

        # Check if the request was successful
        if response.status_code != 200:
            return jsonify({"error": "Error calling Hugging Face API"}), 500

        # Extract response data
        result = response.json()
        labels = result.get("labels", [])
        explanations = result.get("explanations", [])

        # Return the labels and explanations as a JSON response
        return jsonify({"labels": labels, "explanations": explanations})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while processing the request"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
