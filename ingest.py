# -*- coding: utf-8 -*-
"""
最小向量化脚本（保留 RAG 所需的全部功能）
- 从 data/ 读取 .txt/.md
- 用自定义分隔符 <<<SOC-BLOCK>>> 切块
- SentenceTransformer 产出向量，并在库内完成归一化（无需手写 numpy）
- 建 FAISS 内积索引（等价余弦），保存到 vectorstore/

注意：本文件不需要 numpy，因为：
  1) 归一化由 sentence-transformers 内部完成（normalize_embeddings=True）
  2) 相似度由 FAISS 计算（index.search）
"""
import os
import pickle
import re

import faiss
from sentence_transformers import SentenceTransformer

# ===== 配置 =====
CUSTOM_SEP = r"<<<SOC-BLOCK>>>"
DATA_DIR = "data"
OUT_DIR = "vectorstore"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
MIN_LEN = 40  # 过滤极短噪声块（字符数）

def split_by_sep(text: str, min_len: int = MIN_LEN):
    """允许分隔符两侧有空白，合并连续分隔符；过滤过短块。"""
    blocks = [b.strip() for b in re.split(r"(?:\s*<<<SOC-BLOCK>>>\s*)+", text) if b.strip()]
    return [b for b in blocks if len(b) >= min_len]

# ===== 主流程 =====
os.makedirs(OUT_DIR, exist_ok=True)
print("[1/5] 扫描 data/ ...")

docs, metas = [], []
for fname in os.listdir(DATA_DIR):
    if fname.lower().endswith((".txt", ".md")):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        chunks = split_by_sep(raw, min_len=MIN_LEN)
        for i, blk in enumerate(chunks):
            docs.append(blk)
            metas.append({"path": path, "chunk_id": i, "text": blk})

print(f"找到 {len(metas)} 个 chunk，来自 {DATA_DIR}/ 下的多篇文档")
if not docs:
    print("data/ 下没有可读文本（.txt/.md）")
    raise SystemExit

print("[2/5] 加载嵌入模型（首次会下载，可能需要几分钟）...")
model = SentenceTransformer(EMB_MODEL)

print("[3/5] 生成向量（内部已做归一化）...")
emb = model.encode(
    docs,
    normalize_embeddings=True,   # ✅ 库内归一化，免手写 numpy
    convert_to_numpy=True,
    show_progress_bar=True,
)

print("[4/5] 构建并保存索引 ...")
# 归一化 + 内积 ≈ 余弦相似度；因此用 IndexFlatIP
emb = emb.astype("float32")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(metas, f)

print(f"[5/5] 完成：{len(docs)} 个文档 → 维度 {emb.shape[1]}，已保存到 {OUT_DIR}/")
