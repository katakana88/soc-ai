# -*- coding: utf-8 -*-
"""
RAG 问答
- 向量检索：bge-small-zh + FAISS（内积≈余弦）
- 交叉重排：BAAI/bge-reranker-base（让“最能答题”的片段排前）
- 生成：Qwen2.5-0.5B-Instruct + chat 模板
- 交互模式：终端输入问题即可

依赖：faiss-cpu, sentence-transformers, transformers, torch
"""
import os
import pickle

import faiss
import torch
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- 配置 ----------
VSTORE_DIR = "vectorstore"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"            # 与 ingest 保持一致
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"       
RERANKER_NAME = "BAAI/bge-reranker-base"        
TOPK = 12                                       # 初筛候选数
PER_PASSAGE_CHARS = 500                         # 片段截断长度
MAX_NEW_TOKENS = 400                            # 生成字数
RERANK = True

import re

ZONE_NAME = {
    1: "第一区（中央区·心智控制区）",
    2: "第二区（上方区·自主意识区）",
    3: "第三区（下方区·神性连接区）",
}


# ---------- 加载 ----------
def load_retriever():
    index = faiss.read_index(os.path.join(VSTORE_DIR, "index.faiss"))
    with open(os.path.join(VSTORE_DIR, "meta.pkl"), "rb") as f:
        metas = pickle.load(f)
    embedder = SentenceTransformer(EMB_MODEL)
    return index, metas, embedder

def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model
# ===== 全局单例（供 Flask 复用，避免每次加载模型）=====
_RAG_STATE = None
def init_rag():
    """一次性加载（索引/嵌入/LLM/重排器），供服务端复用。"""
    global _RAG_STATE
    if _RAG_STATE is None:
        index, metas, embedder = load_retriever()
        tok, model = load_llm()
        # 预热 CrossEncoder
        if RERANK:
            _ = get_reranker()
        _RAG_STATE = dict(index=index, metas=metas, embedder=embedder, tok=tok, model=model)
    return _RAG_STATE

def answer_query(query: str, topk: int = TOPK, support_k: int = 5):
    """
    单次问答：检索→（可选）关键词补强→重排→组 Prompt→生成
    返回:answer, hits(含 vec 分/重排分）
    """
    st = init_rag()
    index, metas, embedder, tok, model = st["index"], st["metas"], st["embedder"], st["tok"], st["model"]

    hits = search(query, index, metas, embedder, topk=topk)
    
    if RERANK:
        hits = rerank_passages(query, hits)
            # —— 如果是清单型问题，直接规则化抽取，避免模型漏列 ——
        # —— 清单型问题：直接规则化抽取 ——
    

        # —— 清单类问题：直接规则化抽取（完整标题 + 简要说明）
    if is_list_query(query):
        zone_want = parse_zone_from_query(query)       # 1/2/3 或 None=全部
        texts = [h["text"] for h in hits[:max(20, min(len(hits), 40))]]  # 给足上下文
        zone_points = extract_points_with_desc(texts)

        def fmt_zone(zid: int) -> str:
            pts = zone_points.get(zid, {})
            if not pts:
                return ""
            lines = [f"{i}. {pts[i]}" for i in sorted(pts.keys())]
            return f"{ZONE_NAME.get(zid, f'第{zid}区')}：\n" + "\n".join(lines)

        if zone_want:
            body = fmt_zone(zone_want)
            if body:
                return body, hits
        else:
            bodies = [fmt_zone(z) for z in (1, 2, 3)]
            bodies = [b for b in bodies if b]
            if bodies:
                return "\n\n".join(bodies), hits
        # 如果极端情况下规则化没抓到（比如 TopK 太小/命中不含标题），再退回 LLM 生成


    primary, support = hits[0], hits[1:1+support_k]
    prompt = build_prompt_primary(query, primary, support, per_passage_chars=500)
    answer = generate_with_chat_template(tok, model, prompt, max_new_tokens=500)
    return answer, hits

# 交叉重排器（Query, Passage 成对打分，让“能答题”的段排在最前）
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_NAME)
    return _reranker

# ---------- 检索 ----------
def search(query: str, index, metas, embedder, topk=TOPK):
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idxs = index.search(qv, topk)
    hits = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        hit = dict(metas[idx])
        hit["score"] = float(score)
        hits.append(hit)
    return hits

def rerank_passages(query, passages):
    reranker = get_reranker()
    if not passages: 
        return passages
    pairs = [[query, p["text"]] for p in passages]
    scores = reranker.predict(pairs)  # 分数越大越相关
    for p, sc in zip(passages, scores):
        p["rerank_score"] = float(sc)   # ← 关键：写回
    return sorted(passages, key=lambda p: p.get("rerank_score", 0.0), reverse=True)

def is_list_query(q: str) -> bool:
    """是否为清单/列表类问题"""
    q = q.strip()
    return any(k in q for k in ("有哪些", "列出", "清单", "列表", "全部", "有哪些检测点"))

def parse_zone_from_query(q: str) -> int | None:
    """从问题里解析目标区；未指明返回 None（表示全部）"""
    q = q.strip()
    if any(k in q for k in ("第一区", "第一區", "中央区", "心智控制区")): return 1
    if any(k in q for k in ("第二区", "第二區", "上方区", "自主意识区")): return 2
    if any(k in q for k in ("第三区", "第三區", "下方区", "神性连接区")): return 3
    return None

def extract_points_with_desc(texts: list[str]) -> dict[int, dict[int, str]]:
    """
    通用抽取器：从多段文本里解析“区 → {编号: 标题+简短说明}”
    规则：
      - 支持 “检测点N. 标题...” 或 “第N个检测点：” 两种起始格式；
      - 标题后的多行说明会一直收集，直到遇到空行/新检测点/新分区标题；
      - 分区标题出现时，后续检测点归属该分区；没检测到则归属 zone=0（未识别）。
    返回：
      { zone_id: { point_no: "完整标题 + 简要说明", ... }, ... }
    """
    results: dict[int, dict[int, str]] = {}
    cur_zone = 0

    re_zone   = re.compile(r"第\s*([123])\s*区")
    re_point1 = re.compile(r"检测点\s*(\d+)\.\s*(.+)")
    re_point2 = re.compile(r"第\s*(\d+)\s*个检测点[：:]\s*(.*)")

    for t in texts:
        lines = [line.strip() for line in t.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue

            # 1) 分区标题
            m_zone = re_zone.search(line)
            if m_zone:
                cur_zone = int(m_zone.group(1))
                results.setdefault(cur_zone, {})
                i += 1
                continue

            # 2) 检测点起始（两种写法）
            m1 = re_point1.match(line)
            m2 = re_point2.match(line)
            if m1 or m2:
                no = int(m1.group(1) if m1 else m2.group(1))
                title = (m1.group(2) if m1 else m2.group(2)).strip()
                desc_parts = [title]  # 先放完整标题（可能含 vs）

                # 继续收集后续说明行
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if not nxt:
                        break
                    if re_point1.match(nxt) or re_point2.match(nxt):
                        break
                    if re_zone.search(nxt):
                        break
                    desc_parts.append(nxt)
                    j += 1

                full = " ".join(desc_parts).strip()
                results.setdefault(cur_zone, {})
                results[cur_zone][no] = full
                i = j
                continue

            i += 1

    return results

# ---------- Prompt 组装（主片段优先 + 近因偏置） ----------
def build_prompt_primary(query: str, primary, support: list, per_passage_chars=PER_PASSAGE_CHARS) -> str:
    def clip(s: str):
        return s.strip().replace("\n", " ")[:per_passage_chars]

    sup = "\n\n".join([f"[补充片段{i+1}]\n{clip(p['text'])}" for i, p in enumerate(support)])
    # 主片段放最后（近因偏置）
    ctx = (f"{sup}\n\n[主片段]\n{clip(primary['text'])}" if support
           else f"[主片段]\n{clip(primary['text'])}")

    prompt = (
        "请严格按以下步骤完成任务，仅依据【上下文】作答；若信息不足，请回答“我不确定”。\n"
        "———\n"
        "1) 先给出相关性最强的原段落原文，直答用户问题）。\n"
        "2) 优先引用【主片段】，必要时补充引用【补充片段】中的原句或同义转述，并标注片段编号）。\n"
        "3) 扩展与落地（可按中央区/上方区/下方区的相关“检测点”列2-4条可执行建议）。\n"
        "注意：对于意识强度相关的问题，不要引入上下文之外的外部常识；不要空话套话；不能确定就明确说“不确定”。\n\n"
        f"【上下文】\n{ctx}\n\n"
        f"【问题】{query}\n\n"
        "【回答】"
    )
    return prompt


# ---------- 生成（只解码新token + 显式mask） ----------
def generate_with_chat_template(tokenizer, model, prompt: str,
                                max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, temperature=0.2, top_p=0.9):
    messages = [
        {"role": "system", "content": "你是研究意识强度的专家。"},
        {"role": "user", "content": prompt},
    ]
    enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if torch.cuda.is_available():
        enc = enc.to(model.device)

    attention_mask = torch.ones_like(enc)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=enc,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,            # 默认为 False（更稳）；要更活跃就传 True
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )
    gen_ids = out[:, enc.shape[-1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
import re
# 识别“当前区”的标题行（多种写法都兼容）
_RE_ZONE_HDR = re.compile(
    r"意识强度检测点[（(]?\s*第?\s*([123])\s*区[）)]?[:：]?\s*(?:[^\n]*)", re.IGNORECASE
)

# 两种“检测点”写法都支持：
# A) “检测点7. 陌生人敌视程度vs陌生人友好程度”
# B) “意识强度检测点第1区的第7个检测点：” （这行没标题，通常下一行才有 A）
_RE_POINT_A = re.compile(r"检测点\s*(\d+)\.\s*([^\n：:]+?)(?:\s*vs\s*[^\n：:]+)?\s*(?=：|:|\n|$)")
_RE_POINT_B = re.compile(r"第\s*(\d+)\s*个检测点\s*[：:]", re.IGNORECASE)



# ---------- 入口 ----------
if __name__ == "__main__":
    # 1) 加载
    index, metas, embedder = load_retriever()
    tok, model = load_llm()

    print("\n>>> 本地 RAG（CrossEncoder 重排；focus 风格）。输入问题回车；Ctrl+C 退出。")
    while True:
        try:
            query = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        if not query:
            continue

        # 2) 基于向量的初筛
        hits = search(query, index, metas, embedder, topk=TOPK)

        if not hits:
            print("（未检索到相关片段）")
            continue

        # 3) 交叉重排，确保真正能回答问题的段排第一
        hits = rerank_passages(query, hits)

        # 4) 主片段 + 补充片段
        primary, support = hits[0], hits[1:6]

        # 5) Prompt 组装 + 生成
        prompt = build_prompt_primary(query, primary, support, per_passage_chars=PER_PASSAGE_CHARS)
        answer = generate_with_chat_template(tok, model, prompt, max_new_tokens=500, temperature=0.2, top_p=0.9)

        # 6) 输出 + 命中预览
        print("\n【回答】\n", answer)
        print("\n【命中文档（重排后）】")
        for i, h in enumerate(hits[:10], 1):
            prev = h["text"].replace("\n", " ")[:120]
            print(f"- 排名{i}: {os.path.basename(h['path'])}  score={h['score']:.3f}   | {prev}")
