// Minimal accessible student page (Integration plan v0.3, sections 10 and appendix G).
// Every server string is rendered with textContent. The UI replaces its session only with the returned
// committed SessionView and ignores responses from an older request generation.
"use strict";

const ERRORS = {
  UNAUTHORIZED: "Please sign in again.",
  FORBIDDEN: "You do not have access to this course or session.",
  NOTICE_REQUIRED: "Please accept the current notice before continuing.",
  SESSION_CONFLICT: "This conversation changed elsewhere. Start a new session or retry.",
  SESSION_EXPIRED: "This session has ended. Your next message starts a new one.",
  VERSION_MISMATCH: "The notice changed. Reload the page.",
  IDEMPOTENCY_CONFLICT: "That request conflicted with an earlier one. Please retry.",
  REQUEST_IN_PROGRESS: "Your previous message is still being processed.",
  REQUEST_EXPIRED: "That response is no longer available. Please ask again.",
  REQUEST_CANCELLED: "Cancelled.",
  INPUT_TOO_LONG: "Your message is too long. Please shorten it.",
  INVALID_REQUEST: "The request was not valid.",
  EMPTY_INPUT: "Please type a question.",
  CAPACITY_EXCEEDED: "The tutor is busy. Please try again shortly.",
  RATE_LIMITED: "Please wait for your previous message to finish.",
  SERVICE_UNAVAILABLE: "The tutor is temporarily unavailable. Please use your course material and try again later.",
  MAINTENANCE: "The AI tutor is paused. Use your course material and contact your course lead.",
  EXAM_DISABLED: "The tutor is unavailable during this scheduled assessment.",
  CONTENT_UNAVAILABLE: "This source is no longer available.",
};
const HINTS = {
  answer: "Ask a question about the approved course material. Answers are checked against the sources before you see them.",
  tutor: "Type a topic (for example: stem cells) to get an approved practice question, then \"show explanation\" or \"continue\".",
  quiz: "Type a topic (for example: stem cells) for a multiple-choice question, then answer with one letter.",
};

const $ = (id) => document.getElementById(id);
const state = { course: null, notice: null, session: null, generation: 0, pendingKey: null, lastBody: null };

function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40; b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

async function api(method, path, body, headers = {}) {
  const opts = { method, headers: { ...headers }, credentials: "same-origin" };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  return { ok: res.ok, status: res.status, data };
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function show(id) {
  for (const s of ["login", "notice", "chat"]) $(s).hidden = s !== id;
}

function mode() { return document.querySelector("input[name=mode]:checked").value; }

function addMine(text) {
  const li = el("li", "msg me", text);
  $("log").appendChild(li);
  li.scrollIntoView({ block: "end" });
}

function addError(code) {
  const li = el("li", "msg err", ERRORS[code] || ERRORS.SERVICE_UNAVAILABLE);
  $("log").appendChild(li);
  li.scrollIntoView({ block: "end" });
}

function renderReply(r) {
  const li = el("li", "msg bot" + (r.response_code === "A7" ? " a7" : ""));
  const label = { generated: "Verified answer", fixed: "Tutor message", approved_item: "Approved practice item" }[r.content_type];
  li.appendChild(el("div", "code", `${label} - ${r.response_code}`));
  li.appendChild(el("div", "body", r.body));
  if (r.activity) {
    const a = r.activity;
    li.appendChild(el("div", "question", a.question));
    if (a.options.length) {
      const box = el("div", "options");
      for (const o of a.options) {
        const b = el("button", "secondary", `${o.option_id}. ${o.text}`);
        b.type = "button";
        if (a.phase === "question") b.addEventListener("click", () => send(o.option_id));
        else b.disabled = true;
        box.appendChild(b);
      }
      li.appendChild(box);
    }
    const ctl = el("div", "controls");
    const mk = (t, msg) => { const b = el("button", "secondary", t); b.type = "button"; b.addEventListener("click", () => send(msg)); ctl.appendChild(b); };
    if (a.mode === "tutor" && a.phase === "question") mk("Show explanation", "show explanation");
    if (a.phase === "feedback" || a.mode === "tutor") mk("Next question", "continue");
    li.appendChild(ctl);
  }
  if (r.citations.length) {
    const ul = el("ul", "cites");
    for (const c of r.citations) {
      const item = el("li");
      const a = el("a", null, `[${c.citation_id}] ${c.title} - ${c.locator.label}`);
      a.href = c.url;
      a.addEventListener("click", (ev) => { ev.preventDefault(); openEvidence(c.url); });
      item.appendChild(a);
      ul.appendChild(item);
    }
    li.appendChild(ul);
  }
  if (r.notices.length) {
    const ul = el("ul", "notices");
    for (const n of r.notices) ul.appendChild(el("li", null, n));
    li.appendChild(ul);
  }
  const actions = el("div", "row-actions");
  const rep = el("button", "secondary", "Report a problem");
  rep.type = "button";
  rep.addEventListener("click", async () => {
    const res = await api("POST", `/v1/interactions/${encodeURIComponent(r.request_id)}/reports`,
      { category: "answer_error", message: null });
    rep.disabled = true;
    rep.textContent = res.ok ? "Reported" : "Report not sent";
  });
  actions.appendChild(rep);
  li.appendChild(actions);
  $("log").appendChild(li);
  li.scrollIntoView({ block: "end" });
}

function busy(on, text) {
  $("send").disabled = on;
  $("ask-input").disabled = on;
  $("cancel").hidden = !on;
  $("status").textContent = text || "";
}

async function send(text) {
  text = (text ?? $("ask-input").value).trim();
  if (!text || state.pendingKey) return;
  $("retry").hidden = true;
  addMine(text);
  $("ask-input").value = "";
  await submit({
    text,
    requested_mode: mode(),
    session_id: state.session ? state.session.session_id : null,
    expected_session_revision: state.session ? state.session.revision : null,
    notice_version: state.notice,
  });
}

async function submit(body) {
  const generation = ++state.generation;
  const key = uuid();
  state.pendingKey = key;
  state.lastBody = body;
  busy(true, "Finding approved sources and checking the answer...");
  let res;
  try {
    res = await api("POST", `/v1/courses/${encodeURIComponent(state.course)}/interactions`, body, { "Idempotency-Key": key });
  } catch (_) {
    res = { ok: false, data: { error_code: "SERVICE_UNAVAILABLE" } };
  }
  if (generation !== state.generation) return; // stale: discard
  state.pendingKey = null;
  busy(false);
  if (res.ok) {
    if (res.data.session) state.session = res.data.session;
    renderReply(res.data);
  } else {
    const code = (res.data && res.data.error_code) || "SERVICE_UNAVAILABLE";
    if (code === "SESSION_EXPIRED" || code === "SESSION_CONFLICT") state.session = null;
    if (code === "UNAUTHORIZED") { show("login"); return; }
    if (code === "NOTICE_REQUIRED") { await loadNotice(); return; }
    addError(code);
    if (code !== "REQUEST_CANCELLED") $("retry").hidden = false;
  }
  $("ask-input").focus();
}

async function cancel() {
  if (!state.pendingKey) return;
  const key = state.pendingKey;
  $("status").textContent = "Cancelling...";
  await api("POST", `/v1/courses/${encodeURIComponent(state.course)}/interactions/cancel`, undefined, { "Idempotency-Key": key });
}

async function openEvidence(url) {
  const res = await api("GET", url);
  $("evidence").hidden = false;
  if (!res.ok) {
    $("evidence-meta").textContent = "";
    $("evidence-text").textContent = ERRORS[(res.data && res.data.error_code)] || ERRORS.CONTENT_UNAVAILABLE;
    return;
  }
  const p = res.data;
  $("evidence-meta").textContent = `${p.title} - ${p.locator.label} (version ${p.source_version})`;
  $("evidence-text").textContent = p.text;
  $("evidence-close").focus();
}

async function newSession() {
  if (state.session) {
    const s = state.session;
    await api("DELETE", `/v1/sessions/${encodeURIComponent(s.session_id)}`, { expected_session_revision: s.revision });
  }
  state.session = null;
  state.generation++;
  $("log").replaceChildren();
  $("status").textContent = "New session started.";
}

async function loadNotice() {
  const res = await api("GET", "/v1/notices/current");
  if (!res.ok) { show("login"); return; }
  state.notice = res.data.notice_version;
  if (res.data.accepted) { show("chat"); $("ask-input").focus(); return; }
  $("notice-text").textContent = res.data.text;
  show("notice");
  $("notice-accept").focus();
}

async function boot() {
  const me = await api("GET", "/v1/me");
  if (!me.ok) { show("login"); $("login-user").focus(); return; }
  state.course = me.data.course_id;
  $("profile-tag").textContent = me.data.synthetic_identity
    ? `Local ${me.data.profile} build - development identity - not for real student use` : "";
  await loadNotice();
}

document.addEventListener("DOMContentLoaded", () => {
  $("login-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const res = await api("POST", "/v1/dev/login", { user: $("login-user").value.trim() });
    if (res.ok) await boot(); else $("login-user").setAttribute("aria-invalid", "true");
  });
  $("notice-accept").addEventListener("click", async () => {
    const res = await api("POST", "/v1/notices/accept", { notice_version: state.notice, accepted: true });
    if (res.ok) { show("chat"); $("ask-input").focus(); }
  });
  $("ask-form").addEventListener("submit", (ev) => { ev.preventDefault(); send(); });
  $("ask-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(); }
  });
  $("cancel").addEventListener("click", cancel);
  $("retry").addEventListener("click", () => { $("retry").hidden = true; if (state.lastBody) submit(state.lastBody); });
  $("new-session").addEventListener("click", newSession);
  $("evidence-close").addEventListener("click", () => { $("evidence").hidden = true; $("ask-input").focus(); });
  for (const r of document.querySelectorAll("input[name=mode]")) {
    r.addEventListener("change", () => { $("mode-hint").textContent = HINTS[mode()]; });
  }
  $("mode-hint").textContent = HINTS.answer;
  boot();
});
