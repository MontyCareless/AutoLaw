from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForPreTraining, pipeline

app = Flask(__name__)

# Load LegalBERT model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForPreTraining.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Create a pipeline for mask filling (you can replace this with other tasks as needed)
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Use the pipeline to fill a mask (you can adapt this for other analysis, e.g., classification)
    results = pipe(text)
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
