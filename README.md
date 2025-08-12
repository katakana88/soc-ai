# Spectrum of Consciousness AI (SOC_AI)

A local RAG-based LLM chatbot specializing in " 意識強度SoC (Spectrum of Consciousness)”, built for showcasing AI development skills, knowledge base integration, and frontend-backend integration.

意識強度に関する知識ベースを利用したローカルRAG型LLMチャットボットです。  
AI開発スキル、知識ベース統合、フロントエンドとバックエンドの統合を示すために作成しました。

---

## 🌟 Features / 特徴
- **RAG (Retrieval-Augmented Generation)** with a custom vector store for niche knowledge.  
  ニッチな知識ベース向けにカスタムベクトルストアを利用したRAG実装。
- **Cross-Encoder reranking** for more accurate retrieval.  
  より正確な検索のためのクロスエンコーダー再ランキング。
- **Flexible prompt design** for domain-specific QA.  
  特定分野のQAに対応した柔軟なプロンプト設計。
- **Flask + HTML/CSS/JavaScript frontend** for a clean web UI.  
  FlaskとHTML/CSS/JavaScriptによるシンプルなWeb UI。
- **Interactive top-k & context view** for debugging retrieval.  
  検索結果のtop-kとコンテキスト表示によるデバッグ機能。

---

## 🖥 Tech Stack / 技術スタック
- **Python** (Flask, FAISS, sentence-transformers, transformers)
- **Frontend**: HTML, CSS, JavaScript (fetch API)
- **Embedding Model**: [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- **Reranker**: [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- **LLM**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (可換)

---

## 📂 Project Structure / プロジェクト構成
```
SOC-AI/
├── app/ # Flask backend
├── static/ # CSS, JS
├── templates/ # HTML templates
├── data/ # Knowledge base documents (not uploaded)
├── vectorstore/ # FAISS index & metadata
├── ingest.py # Build vector store from data
├── rag_chat.py # RAG + rerank + prompt logic
├── requirements.txt # Python dependencies
└── README.md # This file
```

## 🚀 Installation & Run / インストールと実行
**1. Clone the repository / リポジトリをクローン**
```bash
git clone https://github.com/globaldsae/soc-ai
cd soc-ai
```

**2. Install dependencies / 依存関係をインストール**
```bash
pip install -r requirements.txt 
```

**3. Prepare knowledge base / ナレッジベース準備**

Place .txt documents into data/ folder.

 (https://awakenology.org/) から収集した資料を data/ フォルダに配置。


**4. Build vector store / ベクトルストアの作成**
```bash
python ingest.py
```
**5. Run the web app / Webアプリ実行**
```bash
python app/server.py
```
Then open: http://127.0.0.1:5000

---

## 🎯 Purpose / 目的
This project was created as a demo to demonstrate:
- Practical AI application development from scratch.
- Integration of niche domain knowledge with LLMs.
- End-to-end system: embedding, retrieval, reranking, prompting, and frontend.

このプロジェクトはデモとして作成されました：
- AIアプリケーション開発の実践スキル。
- 特定分野の知識とLLMの統合。
- エンベディング、検索、再ランキング、プロンプト設計、フロントエンドまでの一貫したシステム。

---

## 📚 Knowledge Base / ナレッジベース
https://awakenology.org/

