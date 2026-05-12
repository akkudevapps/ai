"""
Akku RAG API — v2.0
Endpoints:
  POST /api/chat
  POST /admin/add
  POST /admin/bulk-import
  GET  /admin/docs
  POST /admin/doc/edit
  POST /admin/doc/delete
  GET  /admin/stats
  GET  /admin/history
  GET  /admin/collections
  POST /admin/create-collection
  POST /admin/load-collection
  POST /admin/delete-collection
  POST /admin/import-pdf   (auto-activates if pymupdf installed)
"""

import os, json, uuid, datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import openai

app = Flask(__name__)
CORS(app)

CHROMA_PATH      = "./chroma_db"
HISTORY_FILE     = "./history.jsonl"
DEFAULT_COLL     = "akku_knowledge"
LLM_MODEL        = "gpt-oss:120b-cloud"   # ← உங்கள் model பெயர்

embedding_fn  = DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
llm_client    = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

active_col = DEFAULT_COLL   # server-side default (UI always sends collection explicitly)


# ── helpers ────────────────────────────────────────────────────────

def coll(name=None):
    return chroma_client.get_or_create_collection(
        name or active_col, embedding_function=embedding_fn)

def log(action, detail="", collection=None):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action":     action,
            "detail":     detail[:120],
            "collection": collection or active_col,
        }, ensure_ascii=False) + "\n")


# ── CHAT ───────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    d         = request.json or {}
    query     = d.get("message", "").strip()
    coll_name = d.get("collection", active_col)
    bot_name  = d.get("bot_name",  coll_name)
    n         = int(d.get("n_results", 5))
    if not query:
        return jsonify({"error": "message required"}), 400

    c       = coll(coll_name)
    results = c.query(query_texts=[query], n_results=min(n, c.count() or 1))
    docs    = results["documents"][0] if results["documents"] else []
    context = "\n".join(docs)

    prompt = f"""நீங்கள் {bot_name}-ன் AI உதவியாளர்.
கீழே உள்ள தகவல்களை மட்டும் பயன்படுத்தி, 2-3 வரிகளில் தெளிவாகவும் நட்பாகவும் பதில் சொல்லுங்கள்.
தகவல் இல்லாத கேள்விகளுக்கு: "இது குறித்த தகவல் இல்லை. {bot_name}-ஐ பார்க்கவும்."

தகவல்கள்:
{context}

கேள்வி: {query}
பதில்:"""

    resp   = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = resp.choices[0].message.content.strip()
    log("chat", query[:80], coll_name)
    return jsonify({"reply": answer, "sources_used": len(docs)})


# ── ADD single Q&A ─────────────────────────────────────────────────

@app.route("/admin/add", methods=["POST"])
def add_data():
    d         = request.json or {}
    coll_name = d.get("collection", active_col)
    q = d.get("question", "").strip()
    a = d.get("answer",   "").strip()
    if not q or not a:
        return jsonify({"status":"error","message":"question மற்றும் answer தேவை."}), 400
    c      = coll(coll_name)
    doc_id = uuid.uuid4().hex
    c.add(documents=[f"Q: {q}\nA: {a}"],
          metadatas=[{"source":"manual_qa","question":q}],
          ids=[doc_id])
    log("add_qa", q[:80], coll_name)
    return jsonify({"status":"success","message":"தகவல் வெற்றிகரமாகச் சேர்க்கப்பட்டது!","id":doc_id})


# ── BULK IMPORT ────────────────────────────────────────────────────

@app.route("/admin/bulk-import", methods=["POST"])
def bulk_import():
    d         = request.json or {}
    coll_name = d.get("collection", active_col)
    items     = d.get("items", [])
    if not items:
        return jsonify({"status":"error","message":"items[] required"}), 400
    c = coll(coll_name)
    docs, metas, ids = [], [], []
    for item in items:
        q = item.get("q", item.get("question","")).strip()
        a = item.get("a", item.get("answer",  "")).strip()
        if q and a:
            docs.append(f"Q: {q}\nA: {a}")
            metas.append({"source":"bulk_import","question":q})
            ids.append(uuid.uuid4().hex)
    if not docs:
        return jsonify({"status":"error","message":"valid items இல்லை"}), 400
    for i in range(0, len(docs), 100):
        c.add(documents=docs[i:i+100], metadatas=metas[i:i+100], ids=ids[i:i+100])
    log("bulk_import", f"{len(docs)} items", coll_name)
    return jsonify({"status":"success","message":f"{len(docs)} தகவல்கள் சேர்க்கப்பட்டன!","count":len(docs)})


# ── LIST DOCS ──────────────────────────────────────────────────────

@app.route("/admin/docs", methods=["GET"])
def list_docs():
    coll_name = request.args.get("collection", active_col)
    limit     = int(request.args.get("limit",  20))
    offset    = int(request.args.get("offset",  0))
    search    = request.args.get("search", "").strip()
    c         = coll(coll_name)

    if search and c.count() > 0:
        res   = c.query(query_texts=[search], n_results=min(limit, c.count()))
        docs  = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0]  if res["metadatas"]  else []
        ids_  = res["ids"][0]        if res["ids"]        else []
    else:
        all_  = c.get(include=["documents","metadatas"])
        sl    = slice(offset, offset+limit)
        docs  = (all_["documents"] or [])[sl]
        metas = (all_["metadatas"] or [])[sl]
        ids_  = (all_["ids"]       or [])[sl]

    items = [{"id":i,"text":d,"meta":m} for i,d,m in zip(ids_,docs,metas)]
    return jsonify({"documents":items,"total":c.count()})


# ── EDIT DOC ───────────────────────────────────────────────────────

@app.route("/admin/doc/edit", methods=["POST"])
def edit_doc():
    d         = request.json or {}
    coll_name = d.get("collection", active_col)
    doc_id    = d.get("id","").strip()
    q = d.get("question","").strip()
    a = d.get("answer",  "").strip()
    if not doc_id or not q or not a:
        return jsonify({"status":"error","message":"id, question, answer required"}), 400
    c = coll(coll_name)
    c.update(ids=[doc_id],
             documents=[f"Q: {q}\nA: {a}"],
             metadatas=[{"source":"edited","question":q}])
    log("edit_doc", doc_id[:16], coll_name)
    return jsonify({"status":"success","message":"தகவல் புதுப்பிக்கப்பட்டது!"})


# ── DELETE DOC ─────────────────────────────────────────────────────

@app.route("/admin/doc/delete", methods=["POST"])
def delete_doc():
    d         = request.json or {}
    coll_name = d.get("collection", active_col)
    doc_id    = d.get("id","").strip()
    if not doc_id:
        return jsonify({"status":"error","message":"id required"}), 400
    c = coll(coll_name)
    c.delete(ids=[doc_id])
    log("delete_doc", doc_id[:16], coll_name)
    return jsonify({"status":"success","message":"தகவல் நீக்கப்பட்டது!"})


# ── STATS ──────────────────────────────────────────────────────────

@app.route("/admin/stats", methods=["GET"])
def get_stats():
    coll_name = request.args.get("collection", active_col)
    c         = coll(coll_name)
    count     = c.count()
    all_docs  = c.get(include=["documents"])
    total_len = sum(len(d) for d in (all_docs.get("documents") or []))
    return jsonify({
        "status":            "success",
        "collection_name":   coll_name,
        "document_count":    count,
        "estimated_size_kb": round(total_len/1024, 2),
        "embedding_model":   "all-MiniLM-L6-v2 (DefaultEmbeddingFunction)",
        "llm_model":         LLM_MODEL,
    })


# ── HISTORY ────────────────────────────────────────────────────────

@app.route("/admin/history", methods=["GET"])
def get_history():
    coll_name = request.args.get("collection", active_col)
    limit     = int(request.args.get("limit", 30))
    history   = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,"r",encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("collection") == coll_name:
                        history.append(e)
                except Exception:
                    pass
    return jsonify({"history": history[-limit:]})


# ── COLLECTIONS ────────────────────────────────────────────────────

@app.route("/admin/collections", methods=["GET"])
def list_collections():
    return jsonify({"collections":[{"name":c.name,"count":c.count()}
                                    for c in chroma_client.list_collections()]})

@app.route("/admin/create-collection", methods=["POST"])
def create_collection():
    name = (request.json or {}).get("name","").strip()
    if not name:
        return jsonify({"status":"error","message":"Name required"}), 400
    try:
        chroma_client.create_collection(name, embedding_function=embedding_fn)
        log("create_collection", name, name)
        return jsonify({"status":"success","message":f"Collection '{name}' created."})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400

@app.route("/admin/load-collection", methods=["POST"])
def load_collection():
    global active_col
    name = (request.json or {}).get("name","").strip()
    try:
        chroma_client.get_collection(name, embedding_function=embedding_fn)
        active_col = name
        log("load_collection", name, name)
        return jsonify({"status":"success","message":f"Loaded '{name}'."})
    except Exception:
        return jsonify({"status":"error","message":"Collection not found."}), 404

@app.route("/admin/delete-collection", methods=["POST"])
def delete_collection():
    name = (request.json or {}).get("name","").strip()
    try:
        chroma_client.delete_collection(name)
        log("delete_collection", name)
        return jsonify({"status":"success","message":f"Deleted '{name}'."})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400


# ── PDF IMPORT (auto-activates if pymupdf installed) ───────────────

try:
    import fitz  # pip install pymupdf

    @app.route("/admin/import-pdf", methods=["POST"])
    def import_pdf():
        coll_name = request.form.get("collection", active_col)
        if "file" not in request.files:
            return jsonify({"status":"error","error":"No file uploaded"}), 400
        f       = request.files["file"]
        pdf_doc = fitz.open(stream=f.read(), filetype="pdf")
        c       = coll(coll_name)
        chunks, metas, ids = [], [], []
        for page_num, page in enumerate(pdf_doc):
            text = page.get_text().strip()
            for i in range(0, len(text), 500):
                chunk = text[i:i+500].strip()
                if len(chunk) > 40:
                    chunks.append(chunk)
                    metas.append({"source":f.filename,"page":page_num+1,"type":"pdf_chunk"})
                    ids.append(uuid.uuid4().hex)
        for i in range(0, len(chunks), 100):
            c.add(documents=chunks[i:i+100], metadatas=metas[i:i+100], ids=ids[i:i+100])
        log("import_pdf", f"{f.filename} → {len(chunks)} chunks", coll_name)
        return jsonify({"status":"success","message":f"PDF imported: {len(chunks)} chunks","count":len(chunks)})

except ImportError:
    @app.route("/admin/import-pdf", methods=["POST"])
    def import_pdf():
        return jsonify({"status":"error",
                        "error":"PyMuPDF இல்லை. Run: pip install pymupdf"}), 501


# ── MAIN ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print("\n" + "="*52)
    print("  AkkuAI RAG API v2.0  →  http://localhost:5000")
    print(f"  Default collection : {DEFAULT_COLL}")
    print(f"  LLM model          : {LLM_MODEL}")
    print("="*52 + "\n")
    app.run(debug=True, port=5000)