# app/server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from rag_chat import init_rag, answer_query, RERANK  # 直接复用你的核心

app = Flask(__name__, static_folder="../web", static_url_path="/")
CORS(app)

@app.route("/")
def index_page():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/health")
def health():
    # 确保已初始化（首次可能稍慢）
    init_rag()
    return jsonify({"status": "ok", "rerank": RERANK})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    topk = int(data.get("topk") or 32)
    support_k = int(data.get("support_k") or 5)
    if not query:
        return jsonify({"error": "empty query"}), 400
    try:
        ans, hits = answer_query(query, topk=topk, support_k=support_k)
        # 只返回必要字段，避免前端过载
        hits_view = []
        for h in hits[:10]:
            rr = h.get("rerank_score", None)  # 这里取的是该命中的重排分
            hits_view.append({
                "path": os.path.basename(h["path"]),
                "chunk_id": h.get("chunk_id"),
                "vec": round(float(h.get("score", 0.0)), 3),
                "rerank": (round(float(rr), 3) if rr is not None else None),
                "preview": h["text"].replace("\n", " ")[:160],
            })


          # —— 本次生成用到的上下文（主片段+补充片段）——
        primary = hits[0] if hits else None
        support = hits[1:1+support_k] if len(hits) > 1 else []
        ctx = {
            "primary": None if primary is None else {
                "path": os.path.basename(primary["path"]),
                "chunk_id": primary.get("chunk_id"),
                "vec": round(float(primary.get("score", 0.0)), 3),
                "rerank": round(float(primary.get("rerank_score")), 3) if "rerank_score" in primary else None,
                "text": primary["text"][:1200]
            },
            "support": [{
                "path": os.path.basename(s["path"]),
                "chunk_id": s.get("chunk_id"),
                "vec": round(float(s.get("score", 0.0)), 3),
                "rerank": round(float(s.get("rerank_score")), 3) if "rerank_score" in s else None,
                "text": s["text"][:1200]
            } for s in support]
        }

        return jsonify({"answer": ans, "hits": hits_view,"context": ctx})
    except Exception as e:
        # 生产环境记得不要把堆栈暴露给前端

        
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 提前初始化，减少首问延迟
    init_rag()
    app.run(host="127.0.0.1", port=8000, debug=True)
