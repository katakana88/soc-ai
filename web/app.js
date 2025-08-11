const $ = s => document.querySelector(s);
const messages = $("#messages");
const hitlist  = $("#hitlist");
const form     = $("#form");
const input    = $("#q");
const topk     = $("#topk");
const supportk = $("#supportk");
const pingBtn  = $("#ping");
const ctxbtn   = $("#ctxbtn");
const ctxlist  = $("#ctxlist");
let lastContext = null;  // ← 存最近一次上下文

function renderContext(ctx){
  ctxlist.innerHTML = "";
  if(!ctx){ ctxlist.textContent = "（暂无上下文，请先提问）"; return; }

  // 主片段
  if(ctx.primary){
    const p = document.createElement("div");
    p.className = "ctx";
    p.innerHTML = `<b>[主片段]</b> ${ctx.primary.path} · chunk ${ctx.primary.chunk_id} · <b>vec=${ctx.primary.vec}</b> / <b>rerank=${ctx.primary.rerank ?? "-"}</b>\n${ctx.primary.text}`;
    ctxlist.appendChild(p);
  }
  // 补充片段
  (ctx.support || []).forEach((s, i)=>{
    const div = document.createElement("div");
    div.className = "ctx";
    div.innerHTML = `<b>[补充片段${i+1}]</b> ${s.path} · chunk ${s.chunk_id} · <b>vec=${s.vec}</b> / <b>rerank=${s.rerank ?? "-"}</b>\n${s.text}`;
    ctxlist.appendChild(div);
  });
}

ctxbtn.onclick = ()=> renderContext(lastContext);

// 在发送 /chat 后，接收并保存 context
form.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const q = input.value.trim();
  if(!q) return;
  addMsg(q, "you");
  input.value = "";
  form.querySelector("button").disabled = true;

  try{
    const r = await fetch("/chat", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ query: q, topk: Number(topk.value||32), support_k: Number(supportk.value||5) })
    });
    const j = await r.json();
    if(j.answer){
      addMsg(j.answer, "bot");
      renderHits(j.hits || []);
      lastContext = j.context || null;  // ← 保存
    }else{
      addMsg("⚠️ 出错："+(j.error||"未知错误"), "bot");
    }
  }catch(e){
    addMsg("⚠️ 网络错误："+e.message, "bot");
  }finally{
    form.querySelector("button").disabled = false;
  }
});

function addMsg(text, who){
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function renderHits(items){
  hitlist.innerHTML = "";
  items.forEach((h,i)=>{
    const div = document.createElement("div");
    div.className = "hit";
    div.innerHTML = `<b>#${i+1}</b> ${h.path} · chunk ${h.chunk_id} · <b>vec=${h.vec}</b> / <b>rerank=${h.rerank ?? "-"}</b><br>${h.preview}`;
    hitlist.appendChild(div);
  });
}

pingBtn.onclick = async ()=>{
  pingBtn.disabled = true;
  try{
    const r = await fetch("/health");
    const j = await r.json();
    addMsg(`✅ 健康检查：${JSON.stringify(j)}`, "bot");
  }catch(e){
    addMsg("⚠️ 健康检查失败："+e.message, "bot");
  }finally{
    pingBtn.disabled = false;
  }
};

form.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const q = input.value.trim();
  if(!q) return;
  addMsg(q, "you");
  input.value = "";
  form.querySelector("button").disabled = true;

  try{
    const r = await fetch("/chat", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ query: q, topk: Number(topk.value||32), support_k: Number(supportk.value||5) })
    });
    const j = await r.json();
    if(j.answer){
      addMsg(j.answer, "bot");
      renderHits(j.hits || []);
    }else{
      addMsg("⚠️ 出错："+(j.error||"未知错误"), "bot");
    }
  }catch(e){
    addMsg("⚠️ 网络错误："+e.message, "bot");
  }finally{
    form.querySelector("button").disabled = false;
  }
});
