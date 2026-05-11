# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
import openai

# --- PyMuPDF & PDF import இப்போது வேண்டாம் (install ஆகவில்லை) ---
# import fitz
# import re

app = Flask(__name__)
CORS(app)

# ---- ChromaDB Setup ----
CHROMA_PATH = "./chroma_db"

# 🔹 Embedding: Ollama "nomic-embed-text" (pull செய்திருந்தால்)
# இல்லையெனில் HuggingFaceEmbeddings பயன்படுத்தலாம் (மாற்று வழி)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

# ---- Ollama Client (chat model) ----
# உங்களிடம் உள்ள cloud models-ல் ஏதேனும் ஒன்றைப் பயன்படுத்தவும்
# எடுத்துக்காட்டு: "gpt-oss:120b-cloud", "mistral-large-3:675b-cloud"
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# ---- Endpoints ----
@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.json['message']
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""நீங்கள் akkuapps.in-ன் அதிகாரப்பூர்வ AI உதவியாளர். 
கீழே கொடுக்கப்பட்டுள்ள தகவல்களை மட்டும் பயன்படுத்தி, கேள்விக்கு தெளிவாகவும் நட்பாகவும் பதிலளிக்கவும். 
தகவல்கள் இல்லாத கேள்விகளுக்கு "இது குறித்த தகவல் தற்போது கிடைக்கப்பெறவில்லை. மேலும் விவரங்களுக்கு akkuapps.in-ஐ பார்க்கவும்." எனப் பதிலளிக்கவும்.

தகவல்கள்:
{context}

கேள்வி: {query}
பதில்:"""

    response = client.chat.completions.create(
        model="gpt-oss:120b-cloud",   # 🔁 உங்கள் cloud model பெயரை மாற்றவும்
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return jsonify({"reply": answer, "sources": [d.metadata.get("source", "unknown") for d in docs]})


@app.route('/admin/add', methods=['POST'])
def add_data():
    data = request.json
    if 'question' in data and 'answer' in data:
        text = f"Q: {data['question']}\nA: {data['answer']}"
        vectorstore.add_texts([text], metadatas=[{"source": "manual_qa"}])
        vectorstore.persist()
        return jsonify({"status": "success", "message": "தகவல் வெற்றிகரமாகச் சேர்க்கப்பட்டது!"})
    return jsonify({"status": "error", "message": "question மற்றும் answer தேவை."}), 400


# 🔹 PDF import endpoint - PyMuPDF install ஆன பிறகு uncomment செய்யவும்
# @app.route('/admin/import-pdf', methods=['POST'])
# def import_pdf():
#     ...


if __name__ == '__main__':
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    app.run(debug=True, port=5000)