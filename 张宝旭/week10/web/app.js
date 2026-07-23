/* WAF 知识助手 前端逻辑 */
(function () {
  // ---------- 通用 ----------
  const $ = (id) => document.getElementById(id);
  const escapeHtml = (s) =>
    String(s || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));

  // ---------- 当前知识库 ----------
  const KB_KEY = "wy-current-kb";
  const state = {
    currentKb: localStorage.getItem(KB_KEY) || "",   // 由 /api/kbs 返回校正
    currentUser: null,
    kbs: [],
    defaultKb: "",
  };

  // 给 url 自动附加 ?kb=xxx；如果路径里已有 query string 用 & 拼
  function withKb(path) {
    if (!state.currentKb) return path;
    if (/\bkb=/.test(path)) return path;   // 已经手动带了
    const sep = path.includes("?") ? "&" : "?";
    return path + sep + "kb=" + encodeURIComponent(state.currentKb);
  }

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.body !== undefined) {
      // 跨库的请求把 kb 放进 body，更顺手（POST /api/search /api/chat 等）
      const b = { ...opts.body };
      if (state.currentKb && !b.kb) b.kb = state.currentKb;
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(b);
    }
    if (opts.raw) {
      init.method = opts.method || "POST";
      init.body = opts.raw;
    }
    // GET / DELETE / 没 body 的 POST：把 kb 放 query
    const finalPath = (init.body && opts.body) ? path : withKb(path);
    const resp = await fetch(finalPath, init);
    // 401 -> 强制显示登录遮罩
    if (resp.status === 401) {
      showLoginOverlay();
      return { status: 401, data: { ok: false, error: "请先登录" } };
    }
    let data;
    try { data = await resp.json(); }
    catch { data = { ok: false, error: "返回不是 JSON" }; }
    return { status: resp.status, data };
  }

  // ---------- 主题 ----------
  if (localStorage.getItem("waf-theme") === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  }
  $("btnTheme").onclick = () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    if (dark) {
      document.documentElement.removeAttribute("data-theme");
      localStorage.setItem("waf-theme", "light");
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("waf-theme", "dark");
    }
  };

  // ---------- Tab ----------
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
      $("page-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "kb") refreshKbList();
    };
  });

  // ---------- 顶部模型信息 + 知识库下拉 ----------
  // 启动顺序：先拉 /api/me 确认登录状态，再做其他事
  init();

  async function init() {
    // 0. 检查登录状态
    const meResp = await fetch("/api/me").then((r) => r.json()).catch(() => ({}));
    if (!meResp || !meResp.ok || !meResp.username) {
      // 未登录：显示登录遮罩，禁用主页交互
      showLoginOverlay();
      return;
    }
    // 已登录：显示用户名 + 登出按钮
    state.currentUser = meResp.username;
    $("userTag").textContent = `· ${meResp.username}`;
    $("userTag").style.display = "";
    $("btnLogout").style.display = "";

    // 1. 加载库列表
    const kbsResp = await fetch("/api/kbs").then((r) => r.json()).catch(() => ({}));
    if (kbsResp && kbsResp.ok) {
      state.kbs = kbsResp.kbs || [];
      state.defaultKb = kbsResp.default;
      // 若 localStorage 里的 kb 已不存在，回到 default
      if (!state.kbs.find((k) => k.id === state.currentKb)) {
        state.currentKb = state.defaultKb || (state.kbs[0] && state.kbs[0].id) || "";
        localStorage.setItem(KB_KEY, state.currentKb);
      }
    }
    renderKbSwitcher();

    // 2. 模型信息
    const cfgResp = await fetch("/api/config").then((r) => r.json()).catch(() => ({}));
    if (cfgResp && cfgResp.ok) {
      const tag = cfgResp.has_api_key ? `${cfgResp.model || "未配置模型"}` : "⚠ 未配置 api_key";
      $("modelTag").textContent = `· ${tag}`;
    }

    // 3. 首屏：尝试恢复上次活跃会话；没有就显示欢迎语
    if (messagesEl && !messagesEl.querySelector(".msg")) {
      const lastId = state.currentKb ? localStorage.getItem(CONV_ACTIVE_KEY(state.currentKb)) : null;
      let restored = false;
      if (lastId) {
        const list = loadConvList(state.currentKb);
        const c = list.find((x) => x.id === lastId);
        if (c && c.messages.length) {
          currentConv = c;
          messagesEl.innerHTML = "";
          for (const m of c.messages) {
            replayMessage(m);
            history.push({ role: m.role, content: m.content });
          }
          refreshConvTitle();
          restored = true;
        }
      }
      if (!restored) {
        messagesEl.innerHTML = welcomeHtml();
      }
    }
    refreshKbList();
  }

  function renderKbSwitcher() {
    const sel = $("kbSwitcher");
    if (!sel) return;
    sel.innerHTML = "";
    for (const k of state.kbs) {
      const opt = document.createElement("option");
      opt.value = k.id;
      opt.textContent = k.name || k.id;
      if (k.id === state.currentKb) opt.selected = true;
      sel.appendChild(opt);
    }
    const optNew = document.createElement("option");
    optNew.value = "__new__";
    optNew.textContent = "+ 新建知识库...";
    sel.appendChild(optNew);

    sel.onchange = onKbChange;

    // 删除按钮：只剩一个库时禁用
    const del = $("btnDeleteKb");
    if (del) {
      del.disabled = state.kbs.length <= 1;
      del.title = del.disabled ? "至少要保留一个知识库" : "删除当前知识库";
      del.onclick = onDeleteKb;
    }
  }

  // —— 从显示名生成一个候选 id ——
  // 1) 优先取 name 里的英文/数字片段，按 - 串起来
  // 2) 如果一个都没有（纯中文），用 kb-时间戳
  // 3) id 已存在时自动追加 -2 / -3 ...
  function nameToId(name, existing) {
    let base = "";
    const parts = (name || "").toLowerCase().match(/[a-z0-9]+/g);
    if (parts && parts.length) {
      base = parts.join("-").slice(0, 30);
    } else {
      base = "kb-" + Date.now().toString(36);
    }
    let id = base;
    let i = 2;
    while (existing.has(id)) {
      id = `${base}-${i++}`;
    }
    return id;
  }

  async function onKbChange(ev) {
    const v = ev.target.value;
    if (v === "__new__") {
      ev.target.value = state.currentKb;   // 还原下拉
      const name = (prompt("新知识库名称（如：大模型 FAQ）：") || "").trim();
      if (!name) return;
      // 重名（显示名）防呆
      if (state.kbs.find((k) => (k.name || "") === name)) {
        return alert("已经有同名的知识库了");
      }
      const id = nameToId(name, new Set(state.kbs.map((k) => k.id)));
      const { data } = await api("/api/kbs", { method: "POST", body: { id, name } });
      if (!data.ok) return alert("创建失败：" + (data.error || ""));
      state.kbs.push(data.kb);
      switchTo(id);
      return;
    }
    if (v === state.currentKb) return;
    switchTo(v);
  }

  async function onDeleteKb() {
    const cur = state.kbs.find((k) => k.id === state.currentKb);
    if (!cur) return;
    if (state.kbs.length <= 1) return alert("至少要保留一个知识库");
    const typed = prompt(`删除会清掉【${cur.name}】里所有条目和图片，且无法撤销。\n输入库名 "${cur.name}" 以确认：`);
    if (typed === null) return;
    if (typed.trim() !== cur.name) return alert("输入不一致，已取消");

    const { data } = await api(`/api/kbs?kb=${encodeURIComponent(cur.id)}`, { method: "DELETE" });
    if (!data.ok) return alert("删除失败：" + (data.error || ""));

    state.kbs = state.kbs.filter((k) => k.id !== cur.id);
    // 切到默认库或第一个
    const next = state.defaultKb && state.kbs.find((k) => k.id === state.defaultKb)
      ? state.defaultKb
      : state.kbs[0].id;
    switchTo(next);
  }

  function switchTo(kbId) {
    persistCurrentConv && persistCurrentConv();   // 切库前把当前会话存进去
    state.currentKb = kbId;
    localStorage.setItem(KB_KEY, kbId);
    // 切库时清空对话历史与右侧引用面板
    history.length = 0;
    currentConv = null;
    if (messagesEl) messagesEl.innerHTML = welcomeHtml();
    if (refsEl) refsEl.innerHTML = `<div class="muted small">回答出来后这里会显示用到的知识条目。</div>`;
    // 尝试恢复该库上次活跃的会话
    const lastId = localStorage.getItem(CONV_ACTIVE_KEY(kbId));
    if (lastId) {
      const list = loadConvList(kbId);
      const c = list.find((x) => x.id === lastId);
      if (c && c.messages.length) {
        currentConv = c;
        messagesEl.innerHTML = "";
        for (const m of c.messages) {
          replayMessage(m);
          history.push({ role: m.role, content: m.content });
        }
      }
    }
    refreshConvTitle && refreshConvTitle();
    // 清空知识库右侧详情
    kbCurrent = null;
    if (kbDetailEl) kbDetailEl.innerHTML = `<div class="empty">从左侧选一条，或点"＋ 新增条目"</div>`;
    renderKbSwitcher();
    refreshKbList();
  }

  // ---- 登录遮罩 ----
  function showLoginOverlay() {
    // 禁止主页一切交互
    if (messagesEl) messagesEl.innerHTML = "";
    if (refsEl) refsEl.innerHTML = "";
    const overlay = $("loginOverlay");
    if (overlay) {
      overlay.style.display = "flex";
      const inp = $("loginUsername");
      if (inp) { inp.value = ""; inp.focus(); }
      const pwd = $("loginPassword");
      if (pwd) pwd.value = "";
      const err = $("loginError");
      if (err) { err.style.display = "none"; err.textContent = ""; }
    }
  }

  function hideLoginOverlay() {
    const overlay = $("loginOverlay");
    if (overlay) overlay.style.display = "none";
  }

  // 登录表单提交
  $("loginForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = $("loginUsername").value.trim();
    const password = $("loginPassword").value;
    if (!username || !password) return;
    const err = $("loginError");
    try {
      const resp = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (resp.ok || resp.status === 302) {
        hideLoginOverlay();
        // 重新初始化
        state.currentUser = username;
        $("userTag").textContent = `· ${username}`;
        $("userTag").style.display = "";
        $("btnLogout").style.display = "";
        // 重新拉 KB 列表并渲染
        const kbsResp = await fetch("/api/kbs").then((r) => r.json()).catch(() => ({}));
        if (kbsResp && kbsResp.ok) {
          state.kbs = kbsResp.kbs || [];
          state.defaultKb = kbsResp.default;
          state.currentKb = state.defaultKb || (state.kbs[0] && state.kbs[0].id) || "";
          localStorage.setItem(KB_KEY, state.currentKb);
        }
        renderKbSwitcher();
        const cfgResp = await fetch("/api/config").then((r) => r.json()).catch(() => ({}));
        if (cfgResp && cfgResp.ok) {
          const tag = cfgResp.has_api_key ? `${cfgResp.model || "未配置模型"}` : "⚠ 未配置 api_key";
          $("modelTag").textContent = `· ${tag}`;
        }
        messagesEl.innerHTML = welcomeHtml();
        refreshKbList();
      } else {
        const data = await resp.json().catch(() => ({}));
        err.textContent = data.error || "登录失败";
        err.style.display = "block";
      }
    } catch {
      err.textContent = "网络错误，请重试";
      err.style.display = "block";
    }
  });

  // 登出
  $("btnLogout").onclick = async () => {
    await fetch("/api/logout");
    state.currentUser = null;
    state.kbs = [];
    state.currentKb = "";
    localStorage.removeItem(KB_KEY);
    $("userTag").style.display = "none";
    $("btnLogout").style.display = "none";
    $("modelTag").textContent = "";
    renderKbSwitcher();
    showLoginOverlay();
  };

  function welcomeHtml() {
    const cur = state.kbs.find((k) => k.id === state.currentKb);
    const name = cur ? cur.name : "";
    return `
      <div class="welcome">
        <h2>问点${name ? " " + escapeHtml(name) + " " : ""}相关的问题吧</h2>
        <p class="muted">回答会基于"${escapeHtml(name || "知识库")}"里的内容；没有时模型会直说。</p>
      </div>`;
  }

  // ====================================================================
  //                            问答页
  // ====================================================================
  const messagesEl = $("messages");
  const inputEl = $("input");
  const refsEl = $("refs");
  const topkEl = $("topk");
  const decomposeEl = $("decompose");

  // ---- 会话状态 ----
  // 每条会话: {id, kb, title, createdAt, updatedAt, messages:[{role, content, refs?, confidence?, rewritten_query?}]}
  // 当前活动会话 ID 存 localStorage（按 kb 分别记忆）
  const CONV_LIST_KEY = (kb) => `wy-convs-${kb}`;
  const CONV_ACTIVE_KEY = (kb) => `wy-active-conv-${kb}`;
  let currentConv = null;
  const history = [];     // 仅 {role, content}，发给后端用

  function loadConvList(kb) {
    try { return JSON.parse(localStorage.getItem(CONV_LIST_KEY(kb)) || "[]"); }
    catch { return []; }
  }
  function saveConvList(kb, list) {
    localStorage.setItem(CONV_LIST_KEY(kb), JSON.stringify(list));
  }
  function newConvObj(kb) {
    return {
      id: "c-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 6),
      kb: kb,
      title: "",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
  }
  function persistCurrentConv() {
    if (!currentConv || !state.currentKb) return;
    if (!currentConv.messages.length) return;     // 空会话不入库
    const kb = currentConv.kb;
    const list = loadConvList(kb);
    const idx = list.findIndex((c) => c.id === currentConv.id);
    currentConv.updatedAt = Date.now();
    if (!currentConv.title && currentConv.messages.length) {
      const firstUser = currentConv.messages.find((m) => m.role === "user");
      currentConv.title = (firstUser ? firstUser.content : "(无)").slice(0, 30);
    }
    if (idx >= 0) list[idx] = currentConv;
    else list.unshift(currentConv);
    // 限制最多 100 条
    if (list.length > 100) list.length = 100;
    saveConvList(kb, list);
    localStorage.setItem(CONV_ACTIVE_KEY(kb), currentConv.id);
    refreshConvTitle();
  }
  function refreshConvTitle() {
    const t = $("convTitle");
    if (!t) return;
    if (currentConv && currentConv.title) t.textContent = currentConv.title;
    else t.textContent = "新对话";
  }

  // ---- 历史抽屉 ----
  function openHistory() {
    const drawer = $("historyDrawer");
    if (!drawer) return;
    persistCurrentConv();   // 先把当前的存进去
    renderHistoryList();
    drawer.classList.remove("hidden");
  }
  function closeHistory() {
    const drawer = $("historyDrawer");
    if (drawer) drawer.classList.add("hidden");
  }
  function renderHistoryList() {
    const ul = $("historyList");
    if (!ul) return;
    const list = loadConvList(state.currentKb).slice().sort((a, b) => b.updatedAt - a.updatedAt);
    ul.innerHTML = "";
    if (!list.length) {
      ul.innerHTML = `<li class="empty muted small">还没有会话记录</li>`;
      return;
    }
    for (const c of list) {
      const li = document.createElement("li");
      li.className = "history-item" + (currentConv && c.id === currentConv.id ? " active" : "");
      const t = new Date(c.updatedAt);
      const ts = `${t.getMonth() + 1}/${t.getDate()} ${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}`;
      li.innerHTML = `
        <div class="hi-title">${escapeHtml(c.title || "(空)")}</div>
        <div class="hi-meta muted small">${ts} · ${c.messages.length} 条消息</div>
        <button class="hi-del" title="删除">×</button>`;
      li.onclick = (ev) => {
        if (ev.target.classList.contains("hi-del")) return;
        loadConv(c.id);
        closeHistory();
      };
      li.querySelector(".hi-del").onclick = (ev) => {
        ev.stopPropagation();
        if (!confirm(`删除会话"${c.title || "(空)"}"？`)) return;
        deleteConv(c.id);
        renderHistoryList();
      };
      ul.appendChild(li);
    }
  }
  function loadConv(id) {
    const list = loadConvList(state.currentKb);
    const c = list.find((x) => x.id === id);
    if (!c) return;
    currentConv = c;
    localStorage.setItem(CONV_ACTIVE_KEY(state.currentKb), id);
    history.length = 0;
    messagesEl.innerHTML = "";
    refsEl.innerHTML = `<div class="muted small">回答出来后这里会显示用到的知识条目。</div>`;
    for (const m of c.messages) {
      replayMessage(m);
      history.push({ role: m.role, content: m.content });
    }
    refreshConvTitle();
  }
  function deleteConv(id) {
    const kb = state.currentKb;
    const list = loadConvList(kb).filter((c) => c.id !== id);
    saveConvList(kb, list);
    if (currentConv && currentConv.id === id) {
      newConv();
    }
  }
  function newConv() {
    persistCurrentConv();
    currentConv = newConvObj(state.currentKb);
    history.length = 0;
    messagesEl.innerHTML = welcomeHtml();
    refsEl.innerHTML = `<div class="muted small">回答出来后这里会显示用到的知识条目。</div>`;
    refreshConvTitle();
  }

  function replayMessage(m) {
    if (m.role === "user") {
      appendMsg("user", escapeHtml(m.content));
    } else if (m.role === "assistant") {
      const botEl = appendMsg("bot", "");
      botEl.innerHTML = renderAnswer(m.content || "(空)", m.refs || []);
      if (m.confidence === "low") {
        const tip = document.createElement("div");
        tip.className = "conf-tip conf-low";
        tip.textContent = "⚠ 知识库里没有高度相关的内容，回答可能仅供参考";
        botEl.insertBefore(tip, botEl.firstChild);
      } else if (m.confidence === "medium") {
        const tip = document.createElement("div");
        tip.className = "conf-tip conf-medium";
        tip.textContent = "ℹ 检索相关度一般，请核对是否准确";
        botEl.insertBefore(tip, botEl.firstChild);
      }
      appendImagesUnder(botEl, m.refs || []);
      botEl.dataset.rawAnswer = m.content || "";
      appendCopyButton(botEl);
      if (m.rewritten_query) appendRewriteHint(botEl, m.rewritten_query);
      bindAnswerInteractions(botEl, m.refs || []);
    }
  }

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $("btnSend").onclick = sendMessage;
  $("btnClear").onclick = () => {
    if (currentConv && currentConv.messages.length) {
      if (!confirm("清空当前对话内容？（已保存的历史不受影响）")) return;
    }
    history.length = 0;
    if (currentConv) currentConv.messages = [];
    messagesEl.innerHTML = welcomeHtml();
    refsEl.innerHTML = `<div class="muted small">回答出来后这里会显示用到的知识条目。</div>`;
    refreshConvTitle();
  };
  $("btnNewConv").onclick = newConv;
  $("btnHistory").onclick = openHistory;
  document.querySelectorAll("[data-close-history]").forEach((b) => b.addEventListener("click", closeHistory));

  function appendMsg(role, html, extraClass = "") {
    const w = messagesEl.querySelector(".welcome");
    if (w) w.remove();
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    wrap.innerHTML = `
      <div class="avatar">${role === "user" ? "我" : "AI"}</div>
      <div class="bubble ${extraClass}">${html}</div>`;
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return wrap.querySelector(".bubble");
  }

  function appendRewriteHint(bubble, rewritten) {
    const div = document.createElement("div");
    div.className = "rewrite-hint";
    div.textContent = `↻ 上下文改写：${rewritten}`;
    bubble.appendChild(div);
  }

  async function sendMessage() {
    const q = inputEl.value.trim();
    if (!q) return;
    if (!currentConv) currentConv = newConvObj(state.currentKb);
    inputEl.value = "";
    appendMsg("user", escapeHtml(q));
    const botEl = appendMsg("bot", '<span class="typing">正在检索知识库并思考</span>');
    history.push({ role: "user", content: q });
    currentConv.messages.push({ role: "user", content: q });

    const { data, status } = await api("/api/chat", {
      method: "POST",
      body: { query: q, history, top_k: parseInt(topkEl.value, 10), decompose: decomposeEl.checked },
    });
    if (data.ok) {
      botEl.innerHTML = renderAnswer(data.answer || "(空)", data.refs || []);
      // 低置信度提示（顶部小标签）
      if (data.confidence === "low") {
        const tip = document.createElement("div");
        tip.className = "conf-tip conf-low";
        tip.textContent = "⚠ 知识库里没有高度相关的内容，回答可能仅供参考";
        botEl.insertBefore(tip, botEl.firstChild);
      } else if (data.confidence === "medium") {
        const tip = document.createElement("div");
        tip.className = "conf-tip conf-medium";
        tip.textContent = "ℹ 检索相关度一般，请核对是否准确";
        botEl.insertBefore(tip, botEl.firstChild);
      }
      // 把最相关的 refs 里的图片渲染到回答下方
      appendImagesUnder(botEl, data.refs || []);
      // 把模型原文存到 dataset 上，复制时用纯净版本
      botEl.dataset.rawAnswer = data.answer || "";
      appendCopyButton(botEl);
      if (data.rewritten_query) appendRewriteHint(botEl, data.rewritten_query);
      history.push({ role: "assistant", content: data.answer || "" });
      currentConv.messages.push({
        role: "assistant",
        content: data.answer || "",
        refs: data.refs || [],
        confidence: data.confidence,
        rewritten_query: data.rewritten_query,
      });
      persistCurrentConv();
      renderRefs(data.refs || []);
      bindAnswerInteractions(botEl, data.refs || []);
    } else {
      // 错误信息支持换行；提供"重试"按钮
      const errText = String(data.error || ("HTTP " + status));
      const detail = data.detail ? String(data.detail).slice(0, 1000) : "";
      botEl.innerHTML =
        `<div class="err">⚠ ${escapeHtml(errText).replace(/\n/g, "<br/>")}</div>` +
        (detail ? `<details class="err-detail"><summary>上游返回详情</summary><pre>${escapeHtml(detail)}</pre></details>` : "") +
        `<div class="err-actions"><button class="ghost retry-btn">重试</button></div>`;
      const retryBtn = botEl.querySelector(".retry-btn");
      if (retryBtn) {
        retryBtn.onclick = () => {
          // 把刚才的用户气泡和这条 bot 气泡都移除，重发一次
          botEl.parentElement && botEl.parentElement.remove();
          const last = messagesEl.querySelector(".msg.user:last-of-type");
          if (last) last.remove();
          // history 里的最后一条 user 也撤销，避免重发时被当上下文重复
          if (history.length && history[history.length - 1].role === "user") history.pop();
          inputEl.value = q;
          sendMessage();
        };
      }
      renderRefs(data.refs || []);
    }
  }

  function renderAnswer(text, refs) {
    // 先按 Markdown 渲染（已做转义，安全），再处理我们自己的 [#id-pN] 角标和 《标题》。
    let html = renderMarkdown(text || "");

    // 1. 把 [#id-pN] 标记替换成可点击角标 [n]，hover 显示原文
    //    多个连写如 [#9-p3][#9-p5] 会变成 [1][2]
    const segIndex = buildSegIndex(refs || []);
    let nextNo = 1;
    const seenKey = new Map();   // "id-pN" -> 显示编号
    html = html.replace(/\[#(\d+)-p(\d+)\]/g, (_, idStr, pidStr) => {
      const key = `${idStr}-p${pidStr}`;
      let no = seenKey.get(key);
      if (no === undefined) { no = nextNo++; seenKey.set(key, no); }
      const seg = segIndex[key];
      const tipText = seg ? seg.text.replace(/"/g, "&quot;") : "未在引用中找到该段";
      const anchor = seg ? seg.anchor : "";
      const cls = seg ? "cite" : "cite cite-missing";
      return `<a class="${cls}" data-no="${no}" data-key="${key}" data-anchor="${anchor}" data-pid="p${pidStr}" title="${escapeHtml(tipText)}">[${no}]</a>`;
    });
    // 2. 《条目标题》也仍然变成可点击的引用
    html = html.replace(/《([^》]+)》/g, (_, t) => `<a class="ref-link" data-title="${escapeHtml(t)}">《${escapeHtml(t)}》</a>`);
    return html;
  }

  // 极简 Markdown -> HTML（够用就行：粗体/斜体/代码/标题/列表/链接/引用/水平线）
  // 输入是用户/模型的原始文本，输出是 HTML 字符串（已对未识别处的 < > 等转义）
  function renderMarkdown(src) {
    src = String(src || "");
    // —— 1. 提取代码块 ``` ```，先用占位符保护，避免里面的 ** _ # 被解析
    const codeBlocks = [];
    src = src.replace(/```([a-zA-Z0-9_+\-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const safe = escapeHtml(code.replace(/\n$/, ""));
      const cls = lang ? ` class="lang-${escapeHtml(lang)}"` : "";
      codeBlocks.push(`<pre><code${cls}>${safe}</code></pre>`);
      return `\u0000CB${codeBlocks.length - 1}\u0000`;
    });
    // —— 2. 提取行内代码 `xxx`
    const inlineCodes = [];
    src = src.replace(/`([^`\n]+)`/g, (_, code) => {
      inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
      return `\u0000IC${inlineCodes.length - 1}\u0000`;
    });
    // —— 3. 全局转义剩下的内容
    src = escapeHtml(src);
    // —— 4. 块级处理：按双换行切段
    const blocks = src.split(/\n{2,}/).map((blk) => renderBlock(blk));
    let html = blocks.join("\n");
    // —— 5. 还原占位符
    html = html.replace(/\u0000CB(\d+)\u0000/g, (_, i) => codeBlocks[+i]);
    html = html.replace(/\u0000IC(\d+)\u0000/g, (_, i) => inlineCodes[+i]);
    return html;
  }

  function renderBlock(blk) {
    blk = blk.replace(/^\n+|\n+$/g, "");
    if (!blk) return "";
    // 代码块占位符独占一段
    if (/^\u0000CB\d+\u0000$/.test(blk)) return blk;
    // 水平线 --- 或 ***
    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(blk)) return "<hr/>";
    // 标题 #..#####
    const mh = blk.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (mh && !blk.includes("\n")) {
      const level = Math.min(6, mh[1].length);
      // 在气泡里的标题级别都偏小，从 h4 起步
      const tag = "h" + Math.max(4, level + 2);
      return `<${tag}>${renderInline(mh[2])}</${tag}>`;
    }
    // 引用块 > xxx
    if (/^\s*>\s/.test(blk)) {
      const inner = blk.split("\n").map((l) => l.replace(/^\s*>\s?/, "")).join("\n");
      return `<blockquote>${renderInline(inner).replace(/\n/g, "<br/>")}</blockquote>`;
    }
    // 列表（无序 / 有序），允许混合短行
    const lines = blk.split("\n");
    const isUL = lines.every((l) => /^\s*[-*+]\s+/.test(l));
    const isOL = lines.every((l) => /^\s*\d+\.\s+/.test(l));
    if (isUL || isOL) {
      const tag = isUL ? "ul" : "ol";
      const items = lines
        .map((l) => l.replace(/^\s*(?:[-*+]|\d+\.)\s+/, ""))
        .map((l) => `<li>${renderInline(l)}</li>`)
        .join("");
      return `<${tag}>${items}</${tag}>`;
    }
    // 普通段落：段内换行用 <br/>
    return `<p>${renderInline(blk).replace(/\n/g, "<br/>")}</p>`;
  }

  function renderInline(s) {
    // 链接 [text](url)
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (_, t, u) => `<a href="${u}" target="_blank" rel="noopener">${t}</a>`);
    // 加粗 **xx** 或 __xx__
    s = s.replace(/\*\*([^\s*][^*]*?[^\s*]|\S)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__([^\s_][^_]*?[^\s_]|\S)__/g, "<strong>$1</strong>");
    // 斜体 *xx* 或 _xx_（避免吃到上面已处理的 strong 内容）
    s = s.replace(/(^|[^*])\*([^\s*][^*]*?[^\s*]|\S)\*(?!\*)/g, "$1<em>$2</em>");
    s = s.replace(/(^|[^_])_([^\s_][^_]*?[^\s_]|\S)_(?!_)/g, "$1<em>$2</em>");
    // 删除线 ~~xx~~
    s = s.replace(/~~([^~]+)~~/g, "<del>$1</del>");
    return s;
  }

  // refs -> { "id-pN": {text, anchor, title, no} }
  function buildSegIndex(refs) {
    const idx = {};
    for (const r of refs || []) {
      for (const s of (r.segments || [])) {
        idx[`${r.id}-p${s.pid}`] = { text: s.text, anchor: r.anchor, title: r.title, pid: s.pid };
      }
    }
    return idx;
  }

  // 给气泡里的角标 / 标题链接绑事件
  function bindAnswerInteractions(bubble, refs) {
    bubble.querySelectorAll(".ref-link").forEach((a) => {
      a.onclick = () => jumpToKbByTitle(a.dataset.title);
    });
    bubble.querySelectorAll(".cite").forEach((a) => {
      if (!a.dataset.anchor) return;  // 失效的引用不绑跳转
      a.onclick = () => jumpToKbAndHighlight(a.dataset.anchor, a.dataset.pid);
    });
  }

  function renderRefs(refs) {
    if (!refs.length) {
      refsEl.innerHTML = `<div class="muted small">没有命中知识库（这次回答没有引用）。</div>`;
      return;
    }
    refsEl.innerHTML = "";
    refs.forEach((r) => {
      const item = document.createElement("div");
      item.className = "ref-item";
      const thumbs = (r.images || []).slice(0, 4).map((u) =>
        `<img src="${u}" alt="" loading="lazy" />`
      ).join("");
      item.innerHTML = `
        <div class="h"><span>${escapeHtml(r.title)}</span><span class="score">·#${r.id} 命中 ${r.score}</span></div>
        ${thumbs ? `<div class="thumbs">${thumbs}</div>` : ""}
        <div class="body muted small">点击查看原文${(r.images && r.images.length) ? "（含 " + r.images.length + " 张图）" : ""}</div>`;
      // 缩略图点击放大，不要触发跳转
      item.querySelectorAll(".thumbs img").forEach((img) => {
        img.onclick = (ev) => { ev.stopPropagation(); showLightbox(img.src); };
      });
      item.onclick = () => jumpToKbByAnchor(r.anchor);
      refsEl.appendChild(item);
    });
  }

  // 把回答原文里的 [#9-p3] 这种引用标记去掉，作为复制用的"纯净文本"
  function stripCitations(text) {
    if (!text) return "";
    return String(text)
      .replace(/\s*\[#\d+-p\d+\](?:\[#\d+-p\d+\])*/g, "")  // 连写多个角标
      .replace(/[ \t]+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  // 把模型常见的 markdown 标记剥掉（**/__ 加粗、*/_ 斜体、`代码`、~~删除线、
  // ### 标题、> 引用、- / 1. 列表标记、链接 [text](url)、行内 ```代码```）
  function stripMarkdown(text) {
    if (!text) return "";
    let s = String(text);
    // 代码块 ```lang\n...\n``` -> 内部纯文本
    s = s.replace(/```[a-zA-Z0-9_+\-]*\n?([\s\S]*?)```/g, (_, code) => code.trim());
    // 行内代码 `xx` -> xx
    s = s.replace(/`([^`\n]+)`/g, "$1");
    // 链接 [text](url) -> text (url)
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, "$1 ($2)");
    // 加粗 / 斜体（先处理 **/__，再处理 */_ 避免吃错）
    s = s.replace(/\*\*([^\s*][^*]*?[^\s*]|\S)\*\*/g, "$1");
    s = s.replace(/__([^\s_][^_]*?[^\s_]|\S)__/g, "$1");
    s = s.replace(/(^|[^*])\*([^\s*][^*]*?[^\s*]|\S)\*(?!\*)/g, "$1$2");
    s = s.replace(/(^|[^_])_([^\s_][^_]*?[^\s_]|\S)_(?!_)/g, "$1$2");
    // 删除线 ~~xx~~
    s = s.replace(/~~([^~]+)~~/g, "$1");
    // 标题前的 #
    s = s.replace(/^\s{0,3}#{1,6}\s+/gm, "");
    // 引用块前的 >
    s = s.replace(/^\s*>\s?/gm, "");
    // 列表项前的 - * + 或 1.
    s = s.replace(/^\s*[-*+]\s+/gm, "• ");
    s = s.replace(/^\s*\d+\.\s+/gm, (m) => m.trim() + " ");
    // 水平线
    s = s.replace(/^\s*(?:[-*_]\s*){3,}\s*$/gm, "");
    return s;
  }

  function toCleanText(rawAnswer) {
    return stripMarkdown(stripCitations(rawAnswer));
  }

  // 给 AI 回答气泡加一个"复制"按钮，点了复制纯净文本到剪贴板
  function appendCopyButton(bubble) {
    const bar = document.createElement("div");
    bar.className = "answer-actions";
    bar.innerHTML = `<button class="copy-btn ghost" title="复制纯净文本">📋 复制</button>`;
    bubble.appendChild(bar);
    const btn = bar.querySelector(".copy-btn");
    btn.onclick = async () => {
      const raw = bubble.dataset.rawAnswer || "";
      const clean = toCleanText(raw);
      try {
        await navigator.clipboard.writeText(clean);
        const orig = btn.textContent;
        btn.textContent = "✓ 已复制";
        setTimeout(() => { btn.textContent = orig; }, 1200);
      } catch {
        // 老浏览器降级
        const ta = document.createElement("textarea");
        ta.value = clean;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        btn.textContent = "✓ 已复制";
        setTimeout(() => { btn.textContent = "📋 复制"; }, 1200);
      }
    };
  }

  // 拖选复制时（Ctrl+C / 右键复制）也自动剥掉 [1] [2] 角标
  document.addEventListener("copy", (ev) => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;
    // 仅在选区落在 AI 气泡里时介入
    let node = sel.anchorNode;
    while (node && node !== document.body) {
      if (node.nodeType === 1 && node.classList && node.classList.contains("bubble")) break;
      node = node.parentNode;
    }
    if (!node || node === document.body) return;

    // 拿选区文本，干掉 [1] [2] 这种角标的可见文字
    let text = sel.toString();
    text = text.replace(/\[\d+\](?:\[\d+\])*/g, "");
    text = text.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n");
    if (ev.clipboardData) {
      ev.clipboardData.setData("text/plain", text);
      ev.preventDefault();
    }
  });

  // 把命中条目里的图片渲染到回答气泡下方
  function appendImagesUnder(bubble, refs) {
    // 取前 2 条 ref 的图，最多展示 6 张
    const pics = [];
    for (const r of refs.slice(0, 2)) {
      for (const u of (r.images || [])) {
        pics.push({ url: u, title: r.title, anchor: r.anchor });
        if (pics.length >= 6) break;
      }
      if (pics.length >= 6) break;
    }
    if (!pics.length) return;
    const wrap = document.createElement("div");
    wrap.className = "answer-images";
    wrap.innerHTML = `<div class="ai-title">📷 来自 ${refs.slice(0, 2).map((r) => `《${escapeHtml(r.title)}》`).join("、")} 的截图</div>` +
      `<div class="ai-grid">` +
      pics.map((p) => `<img src="${p.url}" alt="" loading="lazy" data-anchor="${p.anchor}" />`).join("") +
      `</div>`;
    bubble.appendChild(wrap);
    wrap.querySelectorAll("img").forEach((img) => {
      img.onclick = () => showLightbox(img.src);
    });
  }

  // ====================================================================
  //                           知识库页
  // ====================================================================
  const kbListEl = $("kbList");
  const kbDetailEl = $("kbDetail");
  const kbStatEl = $("kbStat");
  const kbSearchEl = $("kbSearch");
  let kbAll = [];          // slim 列表（id/anchor/title/tags）
  let kbFull = null;       // 含 plain 的全量缓存：null 未加载，array 已加载
  let kbFullKb = null;     // kbFull 对应的库 id（切库后失效）
  let kbCurrent = null;
  let kbHighlightWords = [];  // 详情页要高亮的关键词（搜索时设置）

  // 输入时即时过滤；首次输入时懒加载 full 数据
  kbSearchEl.addEventListener("input", onKbSearch);

  async function onKbSearch() {
    const kw = kbSearchEl.value.trim();
    if (kw && (!kbFull || kbFullKb !== state.currentKb)) {
      // 懒加载：第一次输入搜索词时才拉全量数据
      kbStatEl.textContent = "正在加载正文以支持全文搜索...";
      try {
        const { data } = await api("/api/entries-full");
        if (data.ok) {
          kbFull = data.entries || [];
          kbFullKb = data.kb || state.currentKb;
        }
      } catch (e) {
        // 失败也无所谓，回落到只搜标题
      }
    }
    renderKbList();
  }

  async function refreshKbList() {
    const { data } = await api("/api/entries");
    if (!data.ok) return;
    kbAll = data.entries || [];
    // 切库或重新拉列表后，全量缓存失效
    kbFull = null;
    kbFullKb = null;
    renderKbList();
  }

  // 把多个关键词（空格分隔）拆出来
  function parseKeywords(kw) {
    return kw.split(/\s+/).map((s) => s.trim().toLowerCase()).filter(Boolean);
  }

  // 在 plain 文本里找命中片段：找到第一个命中词位置，截取前后各 ~30 字
  function makeSnippet(plain, words, max = 100) {
    if (!plain || !words.length) return "";
    const text = plain;
    const lower = text.toLowerCase();
    let bestPos = -1;
    let bestWord = "";
    for (const w of words) {
      const i = lower.indexOf(w);
      if (i >= 0 && (bestPos < 0 || i < bestPos)) {
        bestPos = i;
        bestWord = w;
      }
    }
    if (bestPos < 0) return "";
    const pad = Math.floor((max - bestWord.length) / 2);
    let start = Math.max(0, bestPos - pad);
    let end = Math.min(text.length, bestPos + bestWord.length + pad);
    let s = text.slice(start, end);
    if (start > 0) s = "…" + s;
    if (end < text.length) s = s + "…";
    return s;
  }

  // 把 words 在 html 安全文本里高亮成 <mark>
  function highlightInText(text, words) {
    let html = escapeHtml(text);
    if (!words.length) return html;
    // 按长度倒序，避免短词覆盖长词
    const sorted = [...words].sort((a, b) => b.length - a.length);
    for (const w of sorted) {
      if (!w) continue;
      const re = new RegExp(escapeRegex(w), "gi");
      html = html.replace(re, (m) => `<mark>${m}</mark>`);
    }
    return html;
  }

  function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // 在已有 DOM 子树里把命中文本包成 <mark>。只动文本节点，不破坏标签结构。
  function highlightInDom(rootEl, words) {
    if (!rootEl || !words || !words.length) return;
    const sorted = [...words].filter(Boolean).sort((a, b) => b.length - a.length);
    const re = new RegExp("(" + sorted.map(escapeRegex).join("|") + ")", "gi");
    const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        // 跳过已经在 mark 里的、空白文本节点、script/style 内的
        if (!n.nodeValue || !n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        let p = n.parentNode;
        while (p && p !== rootEl) {
          const tag = p.nodeName ? p.nodeName.toLowerCase() : "";
          if (tag === "mark" || tag === "script" || tag === "style") return NodeFilter.FILTER_REJECT;
          p = p.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const targets = [];
    let cur;
    while ((cur = walker.nextNode())) targets.push(cur);
    for (const node of targets) {
      const txt = node.nodeValue;
      if (!re.test(txt)) { re.lastIndex = 0; continue; }
      re.lastIndex = 0;
      const frag = document.createDocumentFragment();
      let last = 0;
      let m;
      while ((m = re.exec(txt))) {
        if (m.index > last) frag.appendChild(document.createTextNode(txt.slice(last, m.index)));
        const mark = document.createElement("mark");
        mark.textContent = m[0];
        frag.appendChild(mark);
        last = m.index + m[0].length;
      }
      if (last < txt.length) frag.appendChild(document.createTextNode(txt.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  function renderKbList() {
    const kw = kbSearchEl.value.trim().toLowerCase();
    const words = parseKeywords(kw);
    kbHighlightWords = words;

    let list;
    let withSnippets = false;
    if (!kw) {
      list = kbAll.slice();
    } else if (kbFull && kbFullKb === state.currentKb) {
      // 全量搜索（标题 + 标签 + 正文，多关键词全部命中）
      list = kbFull.filter((e) => {
        const hay = (e.title + "\n" + (e.tags || []).join(" ") + "\n" + (e.plain || "")).toLowerCase();
        return words.every((w) => hay.includes(w));
      });
      withSnippets = true;
    } else {
      // 兜底：只搜标题/标签
      list = kbAll.filter((e) =>
        words.every((w) =>
          e.title.toLowerCase().includes(w) ||
          (e.tags || []).some((t) => t.toLowerCase().includes(w))
        )
      );
    }

    kbListEl.innerHTML = "";
    for (const e of list) {
      const li = document.createElement("li");
      li.dataset.anchor = e.anchor;
      if (kbCurrent && kbCurrent.anchor === e.anchor) li.classList.add("active");
      const titleHtml = words.length ? highlightInText(e.title, words) : escapeHtml(e.title);
      let snippetHtml = "";
      if (withSnippets) {
        const s = makeSnippet(e.plain || "", words);
        if (s && !e.title.toLowerCase().includes(words[0])) {
          // 标题里没出现关键词时显示正文片段
          snippetHtml = `<div class="snippet muted small">${highlightInText(s, words)}</div>`;
        }
      }
      li.innerHTML = `
        <div class="row1">
          <span class="id">#${e.id}</span>
        </div>
        <div class="t">${titleHtml}</div>
        ${(e.tags && e.tags.length) ? `<div class="badges">${e.tags.map((t) => `<span class="badge">${escapeHtml(t)}</span>`).join("")}</div>` : ""}
        ${snippetHtml}`;
      li.onclick = () => loadKbDetail(e.anchor);
      kbListEl.appendChild(li);
    }
    const total = kbAll.length, shown = list.length;
    kbStatEl.textContent = kw ? `匹配 ${shown} / 共 ${total}` : `共 ${total} 条`;
  }

  async function loadKbDetail(anchor) {
    const { data } = await api("/api/entry?anchor=" + encodeURIComponent(anchor));
    if (!data.ok) {
      kbDetailEl.innerHTML = `<div class="empty">条目不存在</div>`;
      return;
    }
    kbCurrent = data.entry;
    document.querySelectorAll("#kbList li").forEach((li) =>
      li.classList.toggle("active", li.dataset.anchor === anchor));
    renderKbDetail(kbCurrent);
  }

  function renderKbDetail(entry) {
    const root = document.createElement("div");

    const toolbar = document.createElement("div");
    toolbar.className = "toolbar";
    toolbar.innerHTML = `
      <button class="ghost" id="btnEdit">✎ 编辑</button>
      <button class="ghost" id="btnAskAbout">💬 在问答页提问</button>`;
    root.appendChild(toolbar);

    const h = document.createElement("h1");
    h.textContent = entry.title;
    root.appendChild(h);

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML =
      `<span>#${entry.id}</span>` +
      (entry.tags || []).map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join("");
    root.appendChild(meta);

    if (entry.html) {
      const wrap = document.createElement("div");
      wrap.className = "ql-rendered ql-editor";
      wrap.innerHTML = sanitizeAndFixImg(entry.html);
      // 图片点击放大
      wrap.querySelectorAll("img").forEach((img) => {
        img.onclick = () => showLightbox(img.src);
      });
      // 给段落级元素打上 data-pid，跟后端 entry_segments 顺序保持一致
      assignPidToHtml(wrap);
      root.appendChild(wrap);
    } else {
      let pidCounter = 0;
      const nextPid = () => ++pidCounter;
      for (const b of entry.blocks || []) {
        if (b.type === "h4") {
          const el = document.createElement("h4");
          el.textContent = b.text;
          el.dataset.pid = "p" + nextPid();
          root.appendChild(el);
        } else if (b.type === "p") {
          const el = document.createElement("p");
          el.textContent = b.text;
          el.dataset.pid = "p" + nextPid();
          root.appendChild(el);
        } else if (b.type === "img") {
          const el = document.createElement("img");
          const kb = state.currentKb || "default";
          el.src = b.src.startsWith("images/") ? `/kb/${kb}/` + b.src : b.src;
          el.loading = "lazy";
          el.onclick = () => showLightbox(el.src);
          root.appendChild(el);
        } else if (b.type === "table") {
          const tbl = document.createElement("table");
          for (const row of b.rows || []) {
            const tr = document.createElement("tr");
            tr.dataset.pid = "p" + nextPid();
            for (const c of row) { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); }
            tbl.appendChild(tr);
          }
          root.appendChild(tbl);
        }
      }
    }

    kbDetailEl.innerHTML = "";
    kbDetailEl.appendChild(root);

    // 如果在搜索状态，给详情区里命中关键词加 <mark>
    if (kbHighlightWords && kbHighlightWords.length) {
      highlightInDom(root, kbHighlightWords);
    }

    $("btnEdit").onclick = () => openEditModal(entry);
    $("btnAskAbout").onclick = () => {
      document.querySelector('.tab[data-tab="chat"]').click();
      inputEl.value = `关于"${entry.title}"，请详细说明`;
      inputEl.focus();
    };
  }

  // 把 'images/xxx' 这种相对路径修正成 /kb/<currentKb>/images/xxx
  function sanitizeAndFixImg(html) {
    const kb = state.currentKb || "default";
    return String(html || "").replace(/<img\s+([^>]*?)src="([^"]+)"([^>]*)>/g, (m, pre, src, post) => {
      let fixed = src;
      if (/^images\//.test(src)) fixed = `/kb/${kb}/` + src;
      // 兼容老数据可能存的 /kb/images/xxx → /kb/<currentKb>/images/xxx
      else if (/^\/kb\/images\//.test(src)) fixed = `/kb/${kb}/images/` + src.slice("/kb/images/".length);
      return `<img ${pre}src="${fixed}"${post}>`;
    });
  }

  // 给富文本里的段落元素加 data-pid，编号顺序要和后端 entry_segments 保持一致：
  // 后端切段顺序（伪）：按 <p>/<li>/<h*>/<tr> 在 DOM 树里的出现次序，依次给 p1/p2…
  function assignPidToHtml(rootEl) {
    let pid = 0;
    const candidates = rootEl.querySelectorAll("p, li, h1, h2, h3, h4, h5, h6, tr");
    candidates.forEach((el) => {
      // 后端会过滤掉空白段，这里也跟它一致：纯空白不分配 pid
      if (!el.textContent || !el.textContent.trim()) return;
      pid += 1;
      el.dataset.pid = "p" + pid;
    });
  }

  // ============= 富文本编辑器 =============
  let quill = null;
  function ensureQuill() {
    if (quill) return quill;
    quill = new Quill("#editor", {
      theme: "snow",
      modules: {
        toolbar: {
          container: [
            [{ header: [false, 4, 3, 2] }],
            ["bold", "italic", "underline", "strike"],
            [{ list: "ordered" }, { list: "bullet" }],
            [{ color: [] }, { background: [] }],
            [{ align: [] }],
            ["blockquote", "code-block"],
            ["link", "image"],
            ["clean"],
          ],
          handlers: {
            image: () => uploadFromPicker(),
          },
        },
        clipboard: { matchVisual: false },
      },
    });
    // 拖拽 / 粘贴上传图片
    const editorRoot = document.querySelector(".ql-editor");
    editorRoot.addEventListener("paste", onPasteImage, true);
    editorRoot.addEventListener("drop", onDropImage, true);
    return quill;
  }
  async function uploadFromPicker() {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "image/*";
    inp.onchange = async () => {
      if (inp.files && inp.files[0]) await insertImage(inp.files[0]);
    };
    inp.click();
  }
  async function onPasteImage(e) {
    const items = (e.clipboardData || {}).items || [];
    for (const it of items) {
      if (it.kind === "file" && it.type.startsWith("image/")) {
        e.preventDefault();
        await insertImage(it.getAsFile());
        return;
      }
    }
  }
  async function onDropImage(e) {
    if (!e.dataTransfer || !e.dataTransfer.files || !e.dataTransfer.files.length) return;
    const f = e.dataTransfer.files[0];
    if (!f.type.startsWith("image/")) return;
    e.preventDefault();
    await insertImage(f);
  }
  async function insertImage(file) {
    const fd = new FormData();
    fd.append("file", file, file.name || "image.png");
    const resp = await fetch(withKb("/api/upload"), { method: "POST", body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!data.ok) { alert("图片上传失败：" + (data.error || "")); return; }
    const range = quill.getSelection(true);
    quill.insertEmbed(range.index, "image", data.url, "user");
    quill.setSelection(range.index + 1);
  }

  // ============= 编辑弹层 =============
  $("btnNew").onclick = () => openEditModal(null);
  document.querySelectorAll("[data-close-modal]").forEach((b) => b.addEventListener("click", closeEditModal));
  function openEditModal(entry) {
    $("editTitle").textContent = entry ? "编辑条目" : "新增条目";
    $("editTitleInput").value = entry ? entry.title : "";
    $("editTags").value = entry ? (entry.tags || []).join(", ") : "";
    $("btnDelete").style.display = entry ? "inline-block" : "none";
    $("btnDelete").onclick = entry ? () => deleteEntry(entry) : null;
    $("btnSave").onclick = () => saveEntry(entry);
    $("editModal").classList.remove("hidden");

    setTimeout(() => {
      ensureQuill();
      // 用 setContents 替换现有内容
      if (entry && entry.html) {
        quill.root.innerHTML = sanitizeAndFixImg(entry.html);
      } else if (entry) {
        quill.root.innerHTML = blocksToHtml(entry.blocks || []);
      } else {
        quill.setContents([]);
      }
      $("editTitleInput").focus();
    }, 30);
  }
  function closeEditModal() { $("editModal").classList.add("hidden"); }

  function blocksToHtml(blocks) {
    return blocks.map((b) => {
      if (b.type === "h4") return `<h4>${escapeHtml(b.text)}</h4>`;
      if (b.type === "p") return `<p>${escapeHtml(b.text).replace(/\n/g, "<br/>")}</p>`;
      if (b.type === "img") {
        const kb = state.currentKb || "default";
        const src = b.src.startsWith("images/") ? `/kb/${kb}/` + b.src : b.src;
        return `<p><img src="${src}"/></p>`;
      }
      if (b.type === "table") {
        const rows = (b.rows || []).map((r) => "<tr>" + r.map((c) => `<td>${escapeHtml(c)}</td>`).join("") + "</tr>").join("");
        return `<table>${rows}</table>`;
      }
      return "";
    }).join("");
  }

  async function saveEntry(entry) {
    const title = $("editTitleInput").value.trim();
    const tags = $("editTags").value.split(",").map((s) => s.trim()).filter(Boolean);
    if (!title) return alert("标题不能为空");
    const html = quill ? quill.root.innerHTML : "";
    const body = { anchor: entry ? entry.anchor : null, title, html, tags };
    const { data } = await api("/api/entry", { method: "POST", body });
    if (!data.ok) return alert("保存失败：" + (data.error || ""));
    closeEditModal();
    await refreshKbList();
    loadKbDetail(data.entry.anchor);
  }
  async function deleteEntry(entry) {
    if (!confirm(`确定删除"${entry.title}"？`)) return;
    const { data } = await api("/api/entry?anchor=" + encodeURIComponent(entry.anchor), { method: "DELETE" });
    if (!data.ok) return alert("删除失败：" + (data.error || ""));
    closeEditModal();
    kbCurrent = null;
    kbDetailEl.innerHTML = `<div class="empty">条目已删除</div>`;
    refreshKbList();
  }

  // ============= 导入 / 导出 Word =============
  const fileImport = $("fileImport");
  $("btnImport").onclick = () => fileImport.click();
  fileImport.addEventListener("change", () => {
    if (!fileImport.files || !fileImport.files[0]) return;
    openImportConfirm(fileImport.files[0]);
  });

  function openImportConfirm(file) {
    $("cfFileName").textContent = file.name;
    $("cfTyping").value = "";
    $("btnConfirmImport").disabled = true;
    $("confirmModal").classList.remove("hidden");
    $("cfTyping").focus();

    $("cfTyping").oninput = () => {
      $("btnConfirmImport").disabled = $("cfTyping").value.trim() !== "覆盖";
    };
    $("btnConfirmImport").onclick = async () => {
      $("btnConfirmImport").disabled = true;
      $("btnConfirmImport").textContent = "正在解析…";
      const fd = new FormData();
      fd.append("file", file, file.name);
      const resp = await fetch(withKb("/api/import-docx"), { method: "POST", body: fd });
      const data = await resp.json().catch(() => ({}));
      $("btnConfirmImport").textContent = "确认覆盖";
      if (!data.ok) {
        alert("导入失败：" + (data.error || ""));
        return;
      }
      closeConfirm();
      kbCurrent = null;
      kbDetailEl.innerHTML = `<div class="empty">已导入 ${data.count} 条</div>`;
      await refreshKbList();
      alert(`导入成功，共 ${data.count} 条条目`);
      fileImport.value = "";
    };
  }
  function closeConfirm() {
    $("confirmModal").classList.add("hidden");
    fileImport.value = "";
  }
  document.querySelectorAll("[data-close-confirm]").forEach((b) => b.addEventListener("click", closeConfirm));

  $("btnExport").onclick = () => {
    // 直接走浏览器下载
    window.location.href = withKb("/api/export-docx");
  };

  $("btnReindex").onclick = async () => {
    const btn = $("btnReindex");
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "⟳ 索引中…";
    try {
      const { data } = await api("/api/reindex", { method: "POST", body: {} });
      if (!data.ok) {
        alert("索引失败：" + (data.error || ""));
      } else {
        const s = data.stats || {};
        alert(`索引完成。\n模型：${s.model || "-"}\n新增/更新：${s.added || 0}\n跳过：${s.skipped || 0}\n删除：${s.removed || 0}` + (s.failed ? `\n失败：${s.failed}` : ""));
      }
    } catch (e) {
      alert("索引失败：" + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  };

  // 跳转
  function jumpToKbByAnchor(anchor) {
    document.querySelector('.tab[data-tab="kb"]').click();
    refreshKbList().then(() => loadKbDetail(anchor));
  }

  // 跳到知识库 + 滚动到指定 pid + 临时高亮（黄底 + 左侧竖线，几秒后淡出）
  function jumpToKbAndHighlight(anchor, pid) {
    document.querySelector('.tab[data-tab="kb"]').click();
    refreshKbList().then(() => loadKbDetail(anchor)).then(() => highlightPid(pid));
    // loadKbDetail 不是 Promise 链尾，这里再保险等一下
    setTimeout(() => highlightPid(pid), 200);
  }

  function highlightPid(pid) {
    if (!pid) return;
    const el = kbDetailEl.querySelector(`[data-pid="${pid}"]`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.remove("kb-flash"); // 重新触发动画
    void el.offsetWidth;
    el.classList.add("kb-flash");
    // 5 秒后移掉，但保留左侧竖线提示一会儿
    setTimeout(() => el.classList.remove("kb-flash"), 5000);
  }
  function jumpToKbByTitle(title) {
    document.querySelector('.tab[data-tab="kb"]').click();
    refreshKbList().then(() => {
      const e = kbAll.find((x) => x.title === title);
      if (e) loadKbDetail(e.anchor);
      else { kbSearchEl.value = title; renderKbList(); }
    });
  }

  // lightbox
  const lightbox = document.createElement("div");
  lightbox.id = "lightbox";
  const lbImg = document.createElement("img");
  lightbox.appendChild(lbImg);
  document.body.appendChild(lightbox);
  lightbox.onclick = () => lightbox.classList.remove("show");
  function showLightbox(src) {
    lbImg.src = src;
    lightbox.classList.add("show");
  }
})();
