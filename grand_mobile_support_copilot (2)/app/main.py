from collections import Counter, deque
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from app.fallback import build_fallback_response, build_safe_draft_text
from app.pipeline import SupportCopilotPipeline
from app.schemas import ALLOWED_CATEGORIES, TicketRequest, TriageResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("grand_mobile_copilot.api")

SERVICE_NAME = "grand-mobile-support-copilot"
SERVICE_VERSION = "1.0.0"
REPLY_LANGS = ("ru", "kk", "uz", "en", "es", "pt")

# Кольцевой буфер последних ответов /triage для админ-очереди GET /predictions.
# Сырой текст обращения игрока в буфере не хранится.
RECENT_PREDICTIONS: "deque[Dict[str, Any]]" = deque(maxlen=int(os.getenv("PREDICTIONS_BUFFER", "50")))
_TEMPLATES_CACHE: Dict[str, str] = {}

app = FastAPI(
    title="Grand Mobile Support Copilot API",
    description="Case 03 (Decentrathon 6.0) — Automated Support Ticket Triage & Draft Generation",
    version=SERVICE_VERSION,
)

pipeline = SupportCopilotPipeline()


def _health_payload() -> Dict[str, Any]:
    rules_index = pipeline.rules_index
    if rules_index.load_error:
        return {
            "status": "degraded",
            "error": rules_index.load_error,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "rules_count": 0,
            "sections_count": 0,
        }
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "rules_count": len(rules_index.index),
        "sections_count": len(rules_index.sections),
    }


def _record_prediction(result: Any, ticket_id: Optional[str]) -> None:
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    RECENT_PREDICTIONS.append(
        {
            "ticket_id": ticket_id,
            "received_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Статус сервиса и число загруженных цитируемых правил (всегда HTTP 200)."""
    return _health_payload()


@app.get("/healthz")
async def liveness_probe() -> Dict[str, Any]:
    """Liveness-проба для Docker/Kubernetes: процесс жив и отвечает."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/readyz")
async def readiness_probe() -> Any:
    """Readiness-проба: 200 при загруженном rules.json, 503 в degraded-режиме."""
    payload = _health_payload()
    status_code = 200 if payload["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/triage", response_model=TriageResponse)
async def triage_ticket(payload: Any = Body(default={})) -> Any:
    """
    Triage a single player ticket (complaint or report).

    Никогда не возвращает 4xx/5xx из-за формы тела запроса: некорректный JSON
    (массив, строка, `null`, отсутствующие поля) превращается в безопасный
    детерминированный ответ с needs_human = true.
    """
    body: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    try:
        req = TicketRequest(**body)
        result = pipeline.process_ticket(req)
        _record_prediction(result, req.ticket_id)
        return result
    except Exception as exc:
        logger.error("Unexpected error in /triage, returning safe fallback: %s", exc)
        fallback = build_fallback_response(
            player_text=str(body.get("text", "")),
            kind=str(body.get("kind", "report")),
            server_lang="ru",
            reported_nick=None,
            rule_refs_raw=None,
            attachments=0,
            images_count=0,
            is_spam_or_dup=False,
            rag_items=[],
            is_error_fallback=True,
        )
        _record_prediction(fallback, body.get("ticket_id"))
        return JSONResponse(status_code=200, content=fallback.to_dict())


@app.get("/predictions")
async def list_predictions(limit: int = 50) -> Dict[str, Any]:
    """Последние ответы /triage (новые сначала) для админ-очереди; сырой текст не хранится."""
    max_items = RECENT_PREDICTIONS.maxlen or 50
    safe_limit = max(1, min(50 if limit is None else int(limit), max_items))
    items = list(RECENT_PREDICTIONS)[-safe_limit:][::-1]
    return {"total_buffered": len(RECENT_PREDICTIONS), "limit": safe_limit, "predictions": items}


@app.get("/templates")
async def list_templates(category: Optional[str] = None, lang: Optional[str] = None) -> Any:
    """Банк безопасных шаблонов черновиков по категориям и языкам (бонус §13.2)."""
    if category is not None and category not in ALLOWED_CATEGORIES:
        return JSONResponse(status_code=400, content={"error": f"unknown category: {category}"})
    if lang is not None and lang not in REPLY_LANGS:
        return JSONResponse(status_code=400, content={"error": f"unsupported lang: {lang}"})

    categories = [category] if category else list(ALLOWED_CATEGORIES)
    langs = [lang] if lang else list(REPLY_LANGS)
    templates: List[Dict[str, str]] = []
    for cat in categories:
        for lg in langs:
            key = f"{cat}:{lg}"
            if key not in _TEMPLATES_CACHE:
                text, _ = build_safe_draft_text(
                    category=cat,
                    reply_lang=lg,
                    nickname=None,
                    cited_id=[],
                    has_evidence=False,
                    rag_items=[],
                )
                _TEMPLATES_CACHE[key] = text
            templates.append({"category": cat, "lang": lg, "text": _TEMPLATES_CACHE[key]})
    return {"count": len(templates), "templates": templates}


@app.get("/stats")
async def get_runtime_stats() -> Dict[str, Any]:
    history = pipeline.llm_client.call_history
    dedup_reporters = getattr(pipeline.deduplicator, "tracked_reporters", 0)
    if not history:
        return {
            "total_requests": 0,
            "avg_latency_ms": 0.0,
            "avg_cost_usd": 0.0,
            "dedup_groups": dedup_reporters,
            "predictions_buffered": len(RECENT_PREDICTIONS),
        }
    latencies = sorted(m.latency_ms for m in history)
    n = len(latencies)
    p50 = latencies[int(0.50 * (n - 1))]
    p95 = latencies[int(0.95 * (n - 1))]
    avg_cost = sum(m.cost_usd for m in history) / n
    avg_tokens = sum(m.total_tokens for m in history) / n
    return {
        "total_requests": n,
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "avg_cost_usd": round(avg_cost, 6),
        "avg_tokens": round(avg_tokens, 1),
        "cache_hits": sum(1 for m in history if m.cached),
        "skipped_llm": sum(1 for m in history if m.skipped_llm),
        "dedup_groups": dedup_reporters,
        "predictions_buffered": len(RECENT_PREDICTIONS),
    }


@app.post("/clusters")
async def cluster_templates(tickets: List[Dict[str, Any]] = Body(default=[])) -> Dict[str, Any]:
    """
    Bonus endpoint (Section 13.2): automatic template suggestions by clustering similar requests.
    """
    clusters: Dict[str, Dict[str, Any]] = {}
    for item in tickets:
        resp = pipeline.process_ticket(item)
        key = f"{resp.category}:{resp.draft_reply.lang}"
        if key not in clusters:
            clusters[key] = {
                "cluster_id": key,
                "category": resp.category,
                "lang": resp.draft_reply.lang,
                "count": 0,
                "canonical_template": resp.draft_reply.text,
                "cited_rules": Counter(),
            }
        clusters[key]["count"] += 1
        for c in resp.draft_reply.cited:
            clusters[key]["cited_rules"][c] += 1

    formatted = []
    for k, v in sorted(clusters.items(), key=lambda x: x[1]["count"], reverse=True):
        formatted.append({
            "cluster_id": v["cluster_id"],
            "category": v["category"],
            "lang": v["lang"],
            "count": v["count"],
            "canonical_template": v["canonical_template"],
            "top_cited_rules": [r for r, _ in v["cited_rules"].most_common(3)],
        })
    return {"total_tickets": len(tickets), "clusters": formatted}


@app.post("/admin/feedback")
async def save_admin_feedback(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Bonus endpoint (Section 13.3): saves administrator action ('send', 'edit', 'escalate')
    and edited draft text as a training signal in data/admin_feedback.jsonl.
    """
    feedback_path = os.getenv("ADMIN_FEEDBACK_PATH", "data/admin_feedback.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(feedback_path)), exist_ok=True)
    action = str(payload.get("action", "send")).strip().lower()
    if action not in ("send", "edit", "escalate"):
        action = "send"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_id": payload.get("ticket_id", "manual"),
        "action": action,
        "category": payload.get("category"),
        "priority": payload.get("priority"),
        "original_draft": payload.get("original_draft"),
        "final_draft": payload.get("final_draft"),
        "escalated": action == "escalate" or bool(payload.get("needs_human", False)),
    }
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "saved", "action": action, "record": record}


@app.get("/feedback")
async def list_admin_feedback(limit: int = 50) -> Dict[str, Any]:
    """Последние сохранённые сигналы администраторов (новые сначала), без сырого текста тикета."""
    feedback_path = os.getenv("ADMIN_FEEDBACK_PATH", "data/admin_feedback.jsonl")
    safe_limit = max(1, min(50 if limit is None else int(limit), 500))
    if not os.path.exists(feedback_path):
        return {"path": feedback_path, "total_records": 0, "limit": safe_limit, "records": []}

    tail: "deque[Dict[str, Any]]" = deque(maxlen=safe_limit)
    total = 0
    with open(feedback_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            total += 1
            tail.append(record)
    return {
        "path": feedback_path,
        "total_records": total,
        "limit": safe_limit,
        "records": list(tail)[::-1],
    }


@app.get("/admin", response_class=HTMLResponse)
async def admin_ui() -> str:
    """
    Bonus Admin UI (Section 13.3): ticket queue, draft editor, and 'Отправить' / 'Править' / 'Эскалировать'
    buttons that store administrator edits as training signals.
    """
    return """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Grand Mobile Support Copilot — Интерфейс администратора</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 1040px; margin: 1.5rem auto; padding: 0 1rem; background: #0f172a; color: #f8fafc; }
  h1 { color: #38bdf8; margin-bottom: 0.25rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; margin-top: 1rem; }
  label { display: block; margin-top: 0.65rem; font-size: 0.88rem; color: #cbd5e1; }
  textarea, input, select { width: 100%; box-sizing: border-box; padding: 0.5rem; margin-top: 0.25rem; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; }
  .btn-row { display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap; }
  button { background: #0284c7; color: white; border: none; padding: 0.55rem 1rem; border-radius: 6px; cursor: pointer; font-weight: 600; }
  button.send { background: #16a34a; }
  button.edit { background: #d97706; }
  button.escalate { background: #dc2626; }
  pre { background: #020617; padding: 0.85rem; border-radius: 6px; overflow-x: auto; color: #a7f3d0; font-size: 0.85rem; }
  .queue-item { padding: 0.5rem; border-bottom: 1px solid #334155; cursor: pointer; font-size: 0.9rem; }
  .queue-item:hover { background: #334155; }
</style>
</head>
<body>
  <h1>Grand Mobile Support Copilot — Панель администратора</h1>
  <p>Очередь обращений, генерация черновика (/triage) и сохранение обучающего сигнала (/admin/feedback)</p>
  <div class="grid">
    <div class="card">
      <h3>Очередь тикетов (кликните для загрузки)</h3>
      <div class="queue-item" onclick="loadSample('complaint','ru','Arman_Petrov','2.1 ОП',2,'Arman_Petrov оскорблял меня и мою семью в чате на 3 сервере, скрины есть, 2.1 ОП')">[complaint | ru] Оскорбление в чате (2.1 ОП, 2 скрина)</div>
      <div class="queue-item" onclick="loadSample('complaint','kk','Dias_Khan','2.1',0,'Dias_Khan чатта мені жєне отбасымды ќорлады, 2.1')">[complaint | kk] Қорлау cp1251 без скринов (2.1)</div>
      <div class="queue-item" onclick="loadSample('report','uz','','1.4',0,'Akkauntimni buzishdi, parolim ishlamayapti, kira olmayapman!')">[report | uz] Взлом аккаунта (1.4)</div>
      <div class="queue-item" onclick="loadSample('report','ru','','',0,'Помогите, застрял в текстурах под мостом возле больницы!')">[report | ru] Просьба о помощи (3.1)</div>

      <label>Тип обращения (kind):</label>
      <select id="kind"><option value="complaint">complaint (Жалоба)</option><option value="report">report (Репорт)</option></select>
      <label>Язык сервера (server_lang):</label>
      <select id="server_lang"><option value="ru">ru</option><option value="kk">kk</option><option value="uz">uz</option></select>
      <label>Ник обвиняемого (reported_nick):</label>
      <input id="reported_nick" value="Arman_Petrov" />
      <label>Пункты правил (rule_refs_raw):</label>
      <input id="rule_refs_raw" value="2.1 ОП" />
      <label>Вложения (attachments):</label>
      <input id="attachments" type="number" value="2" />
      <label>Текст обращения игрока:</label>
      <textarea id="text" rows="3">Arman_Petrov оскорблял меня и мою семью в чате на 3 сервере, скрины есть, 2.1 ОП</textarea>
      <div class="btn-row">
        <button onclick="runTriage()">Сгенерировать черновик (/triage)</button>
      </div>
    </div>

    <div class="card">
      <h3>Черновик ответа администратора</h3>
      <label>Текст ответа (доступен для редактирования):</label>
      <textarea id="draft_edit" rows="4" placeholder="Сначала выполните триаж..."></textarea>
      <div class="btn-row">
        <button class="send" onclick="sendFeedback('send')">Отправить</button>
        <button class="edit" onclick="sendFeedback('edit')">Править и сохранить сигнал</button>
        <button class="escalate" onclick="sendFeedback('escalate')">Эскалировать</button>
      </div>
      <p id="fb_status" style="color:#38bdf8;font-size:0.9rem;"></p>
      <h3>JSON-ответ /triage:</h3>
      <pre id="out">Ожидание запроса...</pre>
    </div>
  </div>
<script>
let lastTriage = null;
function loadSample(kind, srv, nick, refs, att, txt) {
  document.getElementById('kind').value = kind;
  document.getElementById('server_lang').value = srv;
  document.getElementById('reported_nick').value = nick;
  document.getElementById('rule_refs_raw').value = refs;
  document.getElementById('attachments').value = att;
  document.getElementById('text').value = txt;
  runTriage();
}
async function runTriage() {
  const payload = {
    kind: document.getElementById('kind').value,
    server_lang: document.getElementById('server_lang').value,
    reported_nick: document.getElementById('reported_nick').value || null,
    rule_refs_raw: document.getElementById('rule_refs_raw').value || null,
    messages: [{
      role: 'player',
      text: document.getElementById('text').value,
      attachments: parseInt(document.getElementById('attachments').value || '0', 10)
    }]
  };
  const res = await fetch('/triage', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  lastTriage = await res.json();
  document.getElementById('draft_edit').value = lastTriage.draft_reply ? lastTriage.draft_reply.text : '';
  document.getElementById('out').textContent = JSON.stringify(lastTriage, null, 2);
}
async function sendFeedback(action) {
  if (!lastTriage) return;
  const payload = {
    action: action,
    category: lastTriage.category,
    priority: lastTriage.priority,
    original_draft: lastTriage.draft_reply.text,
    final_draft: document.getElementById('draft_edit').value,
    needs_human: lastTriage.needs_human
  };
  const res = await fetch('/admin/feedback', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const data = await res.json();
  document.getElementById('fb_status').textContent = 'Сигнал сохранён (' + data.action + ') в ' + data.record.timestamp;
}
</script>
</body>
</html>"""
