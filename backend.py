from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import requests
import numpy as np
from datetime import datetime
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# 🔥 MongoDB Setup
MONGO_URI = "mongodb://localhost:27017/"  # Change if using MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client["one_core_ai"]
chat_collection = db["chat_history"]

# 🔥 Ollama API
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:latest"

# 🔥 Use MongoDB Atlas Vector Search (Set to False for local MongoDB)
use_atlas_vector_search = False

# ✅ Function to compute Cosine Similarity (For local MongoDB)
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ✅ Function to get embeddings (You can replace this with OpenAI or Hugging Face)
def get_embedding(text):
    return np.random.rand(1536).tolist()  # Replace with actual embedding model

# 🔥 Generate AI Response + Save to DB
@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", "guest")  # Default to "guest" if user_id is not provided

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Request to Ollama AI
        payload = {"model": MODEL_NAME, "prompt": prompt}
        response = requests.post(OLLAMA_URL, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch response from Ollama"}), 500

        full_response = []
        for line in response.iter_lines():
            if line:
                try:
                    json_data = line.decode('utf-8')
                    response_obj = json.loads(json_data)
                    if "response" in response_obj:
                        full_response.append(response_obj["response"])
                except Exception as e:
                    print(f"Error parsing JSON: {e}")

        final_response = " ".join(full_response)

        # ✅ Generate embedding for vector search
        embedding = get_embedding(prompt)

        # ✅ Store in MongoDB
        chat_entry = {
            "user_id": user_id,
            "prompt": prompt,
            "response": final_response,
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
            "model": MODEL_NAME
        }
        chat_collection.insert_one(chat_entry)

        return jsonify({"output": final_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🖼️ Image Upload + Description Generation
@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload and generate description using LLaVA or similar model"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Open the image
        img = Image.open(file.stream)
        
        # Convert the image to bytes for sending to the model
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')  # Save as PNG or change format as needed
        img_bytes.seek(0)
        
        # Now send the image to the model
        model_url = "http://localhost:5001/predict"  # Replace with the correct URL of your LLaVA model
        files = {'file': ('image.png', img_bytes, 'image/png')}
        
        # Send the image to the LLaVA model API
        response = requests.post(model_url, files=files)
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to get description from the model"}), 500
        
        data = response.json()
        
        # Assuming the model returns a 'description' field with the caption
        if 'description' in data:
            return jsonify({"description": data['description']})
        else:
            return jsonify({"error": "No description found in the response"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔍 Vector Search API
@app.route('/api/vector-search', methods=['POST'])
def vector_search():
    """Find similar past chat messages based on vector similarity"""
    data = request.json
    query_embedding = data.get("embedding")

    if not query_embedding:
        return jsonify({"error": "Embedding required"}), 400

    if use_atlas_vector_search:
        # ✅ MongoDB Atlas Vector Search Query (Only for Atlas)
        query = {
            "vectorSearch": {
                "index": "embedding_vector",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 5
            }
        }
        results = list(chat_collection.aggregate([query]))

    else:
        # ✅ Local MongoDB Cosine Similarity (Manual Search)
        chat_data = list(chat_collection.find({}, {"_id": 0, "embedding": 1, "prompt": 1, "response": 1}))

        # Compute similarity & sort
        chat_data = sorted(chat_data, key=lambda x: cosine_similarity(query_embedding, x["embedding"]), reverse=True)

        results = chat_data[:5]  # Return top 5 matches

    return jsonify({"results": results})

# 🕰 Get Chat History
@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    """Fetch chat history for a specific user"""
    user_id = request.args.get("user_id", "guest")
    chats = list(chat_collection.find({"user_id": user_id}, {"_id": 0}))
    return jsonify(chats)

# 🔄 Get Available Models
@app.route('/api/tags', methods=['GET'])
def get_models():
    """Fetch available models from Ollama API"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({"error": "Failed to fetch models from Ollama"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
