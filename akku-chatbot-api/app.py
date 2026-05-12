import os, json, datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import openai

app = Flask(__name__)
CORS(app)

CHROMA_PATH = "./chroma_db"
HISTORY_FILE = "./history.jsonl"

# ChromaDB setup
embedding_function = DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Active collection (default)
active_collection_name = "akku_knowledge"

def get_active_collection():
    return chroma_client.get_or_create_collection(
        name=active_collection_name,
        embedding_function=embedding_function
    )

# Ollama client (cloud models)
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# History helper
def log_action(action, detail=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": timestamp, "action": action, "detail": detail, "collection": active_collection_name}) + "\n")

# ---- Endpoints ----

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data['message']
    # Allow overriding collection via body (optional)
    coll_name = data.get('collection', active_collection_name)
    coll = chroma_client.get_or_create_collection(coll_name, embedding_function=embedding_function)
    results = coll.query(query_texts=[query], n_results=3)
    docs = results['documents'][0] if results['documents'] else []
    context = "\n".join(docs)

    prompt = f"""நீங்கள் akkuapps.in-ன் அதிகாரப்பூர்வ AI உதவியாளர். 
கீழே கொடுக்கப்பட்டுள்ள தகவல்களை மட்டும் பயன்படுத்தி, கேள்விக்கு தெளிவாகவும் நட்பாகவும் பதிலளிக்கவும். 
தகவல்கள் இல்லாத கேள்விகளுக்கு "இது குறித்த தகவல் தற்போது கிடைக்கப்பெறவில்லை. மேலும் விவரங்களுக்கு akkuapps.in-ஐ பார்க்கவும்." எனப் பதிலளிக்கவும்.

தகவல்கள்:
{context}

கேள்வி: {query}
பதில்:"""

    response = client.chat.completions.create(
        model="gpt-oss:120b-cloud",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return jsonify({"reply": answer})

@app.route('/admin/add', methods=['POST'])
def add_data():
    data = request.json
    coll_name = data.get('collection', active_collection_name)
    coll = chroma_client.get_or_create_collection(coll_name, embedding_function=embedding_function)
    if 'question' in data and 'answer' in data:
        text = f"Q: {data['question']}\nA: {data['answer']}"
        import uuid
        coll.add(documents=[text], metadatas=[{"source": "manual_qa"}], ids=[uuid.uuid4().hex])
        log_action("add_qa", f"Question: {data['question'][:50]}...")
        return jsonify({"status": "success", "message": "தகவல் வெற்றிகரமாகச் சேர்க்கப்பட்டது!"})
    return jsonify({"status": "error", "message": "question மற்றும் answer தேவை."}), 400

@app.route('/admin/stats', methods=['GET'])
def get_stats():
    coll_name = request.args.get('collection', active_collection_name)
    coll = chroma_client.get_or_create_collection(coll_name, embedding_function=embedding_function)
    count = coll.count()
    # Estimate size
    all_docs = coll.get(include=['documents'])
    total_len = sum(len(d) for d in all_docs.get('documents', []))
    size_kb = total_len / 1024
    return jsonify({
        "status": "success",
        "collection_name": coll_name,
        "document_count": count,
        "estimated_size_kb": round(size_kb, 2),
        "embedding_model": "all-MiniLM-L6-v2"
    })

@app.route('/admin/history', methods=['GET'])
def get_history():
    coll_name = request.args.get('collection', active_collection_name)
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('collection') == coll_name:
                        history.append(entry)
                except:
                    pass
    return jsonify({"history": history[-20:]})  # last 20

@app.route('/admin/collections', methods=['GET'])
def list_collections():
    colls = chroma_client.list_collections()
    result = []
    for c in colls:
        result.append({"name": c.name, "count": c.count()})
    return jsonify({"collections": result})

@app.route('/admin/create-collection', methods=['POST'])
def create_collection():
    name = request.json.get('name', '').strip()
    if not name:
        return jsonify({"status": "error", "message": "Name required"}), 400
    try:
        chroma_client.create_collection(name, embedding_function=embedding_function)
        log_action("create_collection", name)
        return jsonify({"status": "success", "message": f"Collection '{name}' created."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/admin/load-collection', methods=['POST'])
def load_collection():
    global active_collection_name
    name = request.json.get('name', '')
    try:
        # Check existence
        chroma_client.get_collection(name, embedding_function=embedding_function)
        active_collection_name = name
        log_action("load_collection", name)
        return jsonify({"status": "success", "message": f"Loaded '{name}'."})
    except Exception as e:
        return jsonify({"status": "error", "message": "Collection not found."}), 404

@app.route('/admin/delete-collection', methods=['POST'])
def delete_collection():
    name = request.json.get('name', '')
    try:
        chroma_client.delete_collection(name)
        log_action("delete_collection", name)
        return jsonify({"status": "success", "message": f"Deleted '{name}'."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# PDF Import (if PyMuPDF is available) – leave commented for now

if __name__ == '__main__':
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    app.run(debug=True, port=5000)