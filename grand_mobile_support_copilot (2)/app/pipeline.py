import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from app.dedup import Deduplicator
from app.fallback import (
    REPORT_PLAYER_ID_REGEX,
    REPORT_VIOLATION_ABBR_REGEX,
    build_fallback_response,
    check_has_evidence,
    infer_category,
)
from app.langdetect import detect_language
from app.llm_client import ArgonLLMClient, BaseLLMProvider, LLMCallMetrics
from app.normalize import extract_rule_refs, normalize_text
from app.postprocess import postprocess_triage_dict
from app.retriever import BM25Retriever
from app.rules_index import RulesIndex
from app.schemas import TicketRequest, TriageResponse

logger = logging.getLogger("grand_mobile_copilot.pipeline")


class SupportCopilotPipeline:
    def __init__(
        self,
        rules_path: Optional[str] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        self.rules_index = RulesIndex(rules_path=rules_path)
        self.retriever = BM25Retriever(rules_index=self.rules_index)
        self.deduplicator = Deduplicator()
        self.llm_client = ArgonLLMClient(provider=llm_provider)

    def _apply_degraded_guard(self, resp: TriageResponse) -> TriageResponse:
        if self.rules_index.load_error or len(self.rules_index.index) == 0:
            resp.needs_human = True
            if "rules.json не загружен" not in resp.reasons:
                resp.reasons = list(resp.reasons[:2]) + ["rules.json не загружен"]
        return resp

    def extract_player_text_and_attachments(
        self, req: TicketRequest
    ) -> Tuple[str, List[Dict[str, Any]], int]:
        norm_messages: List[Dict[str, Any]] = []
        player_parts: List[str] = []
        msg_attachments_sum = 0

        for msg in req.messages or []:
            role = normalize_text(msg.role) or "player"
            txt = normalize_text(msg.text)
            m_att = max(0, int(msg.attachments or 0))
            msg_attachments_sum += m_att
            m_dict: Dict[str, Any] = {"role": role, "text": txt}
            if m_att > 0:
                m_dict["attachments"] = m_att
            norm_messages.append(m_dict)
            if role == "player" and txt:
                player_parts.append(txt)

        if not player_parts and req.text:
            top_txt = normalize_text(req.text)
            if top_txt:
                player_parts.append(top_txt)
                norm_messages.append({"role": "player", "text": top_txt})

        if not player_parts and norm_messages:
            for m in norm_messages:
                if m["text"] and m["role"] != "admin":
                    player_parts.append(m["text"])

        total_attachments = max(int(req.attachments or 0), msg_attachments_sum)
        return "\n".join(player_parts).strip(), norm_messages, total_attachments

    def process_ticket_with_metrics(
        self,
        ticket_input: Any,
        dedup_override: Optional[bool] = None,
    ) -> Tuple[TriageResponse, LLMCallMetrics]:
        t0 = time.perf_counter()
        try:
            if isinstance(ticket_input, TicketRequest):
                req = ticket_input
            elif isinstance(ticket_input, dict):
                req = TicketRequest(**ticket_input)
            else:
                req = TicketRequest()
        except Exception as exc:
            logger.warning("Malformed TicketRequest, falling back to safe defaults: %s", exc)
            req = TicketRequest()

        player_text, norm_messages, total_attachments = self.extract_player_text_and_attachments(req)
        norm_reported_nick = normalize_text(req.reported_nick) if req.reported_nick else None
        norm_rule_refs_raw = normalize_text(str(req.rule_refs_raw)) if req.rule_refs_raw is not None else None

        detected_lang = detect_language(player_text)

        dup_hint = req.DUPLICATE_HINT if req.DUPLICATE_HINT is not None else req.duplicate_hint
        if str(req.verdict or "").strip().lower() == "duplicate":
            dup_hint = True

        if dedup_override is not None:
            is_spam_or_dup = bool(dedup_override or dup_hint)
            dup_by_hist = is_spam_or_dup
        else:
            is_spam_or_dup, dup_by_hist = self.deduplicator.check_and_record(
                text=player_text,
                reporter_hash=req.reporter_hash,
                created_at=req.created_at,
                duplicate_hint=dup_hint,
            )

        rule_refs = extract_rule_refs(player_text, norm_rule_refs_raw)
        has_ev = check_has_evidence(total_attachments, req.images_count, player_text)
        cat_hint = infer_category(
            text=player_text,
            kind=req.kind,
            rule_refs=rule_refs,
            has_evidence=has_ev,
            is_spam=is_spam_or_dup,
            verdict=req.verdict,
            rule_refs_raw=norm_rule_refs_raw,
        )
        rag_items = self.retriever.retrieve(
            query=f"{player_text} {norm_rule_refs_raw or ''}".strip(),
            rule_refs=rule_refs,
            category_hint=cat_hint,
            has_evidence=has_ev,
            kind=req.kind,
            top_k=4,
            verdict=req.verdict,
        )
        rules_context_str = self.retriever.format_context(rag_items)

        cleaned_for_bypass = re.sub(r"<(?:URL|HANDLE)>|https?://\S+", "", player_text, flags=re.IGNORECASE)
        letters_only = [c for c in cleaned_for_bypass if c.isalpha()]
        has_id_or_violation = (
            bool(rule_refs)
            or bool(REPORT_PLAYER_ID_REGEX.search(cleaned_for_bypass))
            or bool(REPORT_VIOLATION_ABBR_REGEX.search(cleaned_for_bypass))
        )
        if req.kind == "report" and not has_id_or_violation and (detected_lang == "und" or len(letters_only) < 3):
            resp = build_fallback_response(
                player_text=player_text,
                kind=req.kind,
                server_lang=req.server_lang,
                reported_nick=norm_reported_nick,
                rule_refs_raw=norm_rule_refs_raw,
                attachments=total_attachments,
                images_count=req.images_count,
                is_spam_or_dup=True,
                rag_items=rag_items,
                is_error_fallback=False,
                verdict=req.verdict,
            )
            resp = self._apply_degraded_guard(resp)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics = LLMCallMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                cached=False,
                skipped_llm=True,
                provider="trivial-bypass",
            )
            self.llm_client.call_history.append(metrics)
            return resp, metrics

        ticket_payload = {
            "kind": req.kind,
            "messages": [m for m in norm_messages if m.get("role") != "admin"] or norm_messages,
            "attachments": total_attachments,
            "images_count": req.images_count or 0,
            "reported_nick": norm_reported_nick,
            "rule_refs_raw": norm_rule_refs_raw,
            "server_lang": req.server_lang or "ru",
            "region": req.region or "cis",
            "verdict": req.verdict,
            "DUPLICATE_HINT": bool(dup_hint or dup_by_hist),
            # Итоговый флаг спама/дубля (включая текстовый антиспам), чтобы confidence
            # не зависел от того, вычислен ли флаг до пайплайна (batch) или внутри него (HTTP).
            "IS_SPAM_OR_DUP": bool(is_spam_or_dup),
        }

        try:
            raw_llm_dict, metrics = self.llm_client.generate_triage(
                rules_context=rules_context_str,
                ticket_payload=ticket_payload,
                rag_items=rag_items,
            )
            final_resp = postprocess_triage_dict(
                raw_data=raw_llm_dict,
                player_text=player_text,
                kind=req.kind,
                server_lang=req.server_lang,
                region=req.region,
                reported_nick=norm_reported_nick,
                rule_refs_raw=norm_rule_refs_raw,
                attachments=total_attachments,
                images_count=req.images_count,
                is_spam_or_dup_detected=is_spam_or_dup,
                rag_items=rag_items,
                rules_index=self.rules_index,
                verdict=req.verdict,
            )
            final_resp = self._apply_degraded_guard(final_resp)
            metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
            return final_resp, metrics
        except Exception as exc:
            logger.error("LLM failed, activating rule-based fallback: %s", exc)
            fallback_resp = build_fallback_response(
                player_text=player_text,
                kind=req.kind,
                server_lang=req.server_lang,
                reported_nick=norm_reported_nick,
                rule_refs_raw=norm_rule_refs_raw,
                attachments=total_attachments,
                images_count=req.images_count,
                is_spam_or_dup=is_spam_or_dup,
                rag_items=rag_items,
                is_error_fallback=True,
                verdict=req.verdict,
            )
            safe_resp = postprocess_triage_dict(
                raw_data=fallback_resp.to_dict(),
                player_text=player_text,
                kind=req.kind,
                server_lang=req.server_lang,
                region=req.region,
                reported_nick=norm_reported_nick,
                rule_refs_raw=norm_rule_refs_raw,
                attachments=total_attachments,
                images_count=req.images_count,
                is_spam_or_dup_detected=is_spam_or_dup,
                rag_items=rag_items,
                rules_index=self.rules_index,
                verdict=req.verdict,
            )
            safe_resp.needs_human = True
            safe_resp.confidence = min(safe_resp.confidence, 0.35)
            safe_resp = self._apply_degraded_guard(safe_resp)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics = LLMCallMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                cached=False,
                skipped_llm=True,
                provider="fallback",
            )
            self.llm_client.call_history.append(metrics)
            return safe_resp, metrics

    def process_ticket(
        self,
        ticket_input: Any,
        dedup_override: Optional[bool] = None,
    ) -> TriageResponse:
        resp, _ = self.process_ticket_with_metrics(ticket_input, dedup_override=dedup_override)
        return resp
