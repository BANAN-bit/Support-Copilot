import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
from app.fallback import build_fallback_response
from app.rules_index import RuleItem

logger = logging.getLogger("grand_mobile_copilot.llm")


class LLMClientError(Exception):
    """Raised when the LLM provider fails or returns invalid JSON after retry."""


@dataclass
class LLMCallMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool = False
    skipped_llm: bool = False
    provider: str = "argon-4.0"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


FALLBACK_SYSTEM_PROMPT = (
    "Ты — ассистент поддержки игры Grand Mobile. Сделай триаж обращения игрока и подготовь ЧЕРНОВИК "
    "ответа администратора на языке игрока.\n"
    "RULES_CONTEXT — единственный разрешённый источник пунктов правил: цитировать можно только id из него.\n"
    "Запрещено обещать разбан, возврат средств, компенсацию, наказание игрока или сроки.\n"
    "Ответ — строго JSON по схеме {{TICKET_JSON}}.\n\nRULES_CONTEXT:\n{{RULES_CONTEXT}}"
)


class BaseLLMProvider(ABC):
    @abstractmethod
    def complete_json(
        self,
        system_prompt: str,
        ticket_payload: Dict[str, Any],
        rag_items: List[RuleItem],
        temperature: float = 0.1,
        max_tokens: int = 400,
        timeout_sec: float = 4.0,
    ) -> Tuple[Dict[str, Any], int, int]:
        """Return (parsed_json_dict, prompt_tokens, completion_tokens)."""


class OpenAICompatibleArgonProvider(BaseLLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str = "argon-4.0") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete_json(
        self,
        system_prompt: str,
        ticket_payload: Dict[str, Any],
        rag_items: List[RuleItem],
        temperature: float = 0.1,
        max_tokens: int = 400,
        timeout_sec: float = 4.0,
    ) -> Tuple[Dict[str, Any], int, int]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}],
            "temperature": max(0.0, min(0.2, temperature)),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
        if resp.status_code != 200:
            raise LLMClientError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        p_tok = int(usage.get("prompt_tokens") or estimate_tokens(system_prompt))
        c_tok = int(usage.get("completion_tokens") or estimate_tokens(content))
        parsed = _parse_json_strict(content)
        return parsed, p_tok, c_tok


class LocalArgonStructuredProvider(BaseLLMProvider):
    def __init__(self, model: str = "argon-4.0-local") -> None:
        self.model = model

    def complete_json(
        self,
        system_prompt: str,
        ticket_payload: Dict[str, Any],
        rag_items: List[RuleItem],
        temperature: float = 0.1,
        max_tokens: int = 400,
        timeout_sec: float = 4.0,
    ) -> Tuple[Dict[str, Any], int, int]:
        messages = ticket_payload.get("messages", [])
        player_texts = [
            str(m.get("text", ""))
            for m in messages
            if isinstance(m, dict) and m.get("role", "player") == "player"
        ]
        if not player_texts and messages:
            player_texts = [str(m.get("text", "")) for m in messages if isinstance(m, dict)]
        full_text = "\n".join(t for t in player_texts if t).strip()

        triage_obj = build_fallback_response(
            player_text=full_text,
            kind=str(ticket_payload.get("kind", "report")),
            server_lang=ticket_payload.get("server_lang", "ru"),
            reported_nick=ticket_payload.get("reported_nick"),
            rule_refs_raw=ticket_payload.get("rule_refs_raw"),
            attachments=ticket_payload.get("attachments", 0),
            images_count=ticket_payload.get("images_count", 0),
            is_spam_or_dup=bool(
                ticket_payload.get("DUPLICATE_HINT", False) or ticket_payload.get("IS_SPAM_OR_DUP", False)
            ),
            rag_items=rag_items,
            is_error_fallback=False,
            verdict=ticket_payload.get("verdict"),
        )
        out_dict = triage_obj.to_dict()
        serialized = json.dumps(out_dict, ensure_ascii=False, separators=(",", ":"))
        p_tok = min(950, estimate_tokens(serialized) + 420)
        c_tok = min(max_tokens, estimate_tokens(serialized))
        return out_dict, p_tok, c_tok


def _parse_json_strict(raw_content: str) -> Dict[str, Any]:
    if not raw_content or not isinstance(raw_content, str):
        raise LLMClientError("Empty LLM response")
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        raise LLMClientError(f"Invalid JSON from LLM: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMClientError("LLM JSON root must be an object")
    return parsed


class ArgonLLMClient:
    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        prompt_path: str = "prompts/system_prompt.txt",
    ) -> None:
        self.prompt_path = prompt_path
        self.system_prompt_template = self._load_prompt_template()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "400"))
        self.timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "4.0"))
        self.input_cost_per_1k = float(os.getenv("LLM_INPUT_COST_PER_1K", "0.0005"))
        self.output_cost_per_1k = float(os.getenv("LLM_OUTPUT_COST_PER_1K", "0.0015"))
        self.cache_enabled = os.getenv("LLM_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.call_history: List[LLMCallMetrics] = []

        if provider is not None:
            self.provider = provider
        else:
            self.provider = self._init_provider_from_env()

    def _load_prompt_template(self) -> str:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        # Локальный детерминированный провайдер не использует промпт, поэтому его отсутствие
        # (например, при копировании только пакета app/) не должно ронять сервис на старте.
        logger.warning(
            "System prompt not found at %s; using built-in fallback prompt (%d chars)",
            self.prompt_path,
            len(FALLBACK_SYSTEM_PROMPT),
        )
        return FALLBACK_SYSTEM_PROMPT

    def _init_provider_from_env(self) -> BaseLLMProvider:
        prov_type = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "https://api.argon.ai/v1").strip()
        model = os.getenv("LLM_MODEL", "argon-4.0").strip()

        if prov_type == "openai" or (prov_type == "auto" and api_key and api_key != "your_api_key_here"):
            return OpenAICompatibleArgonProvider(base_url=base_url, api_key=api_key, model=model)
        return LocalArgonStructuredProvider(model=model)

    def render_prompt(self, rules_context: str, ticket_payload: Dict[str, Any]) -> str:
        ticket_json = json.dumps(ticket_payload, ensure_ascii=False, separators=(",", ":"))
        prompt = self.system_prompt_template.replace("{{RULES_CONTEXT}}", rules_context)
        prompt = prompt.replace("{{TICKET_JSON}}", ticket_json)
        return prompt

    def _compute_cache_key(self, rules_context: str, ticket_payload: Dict[str, Any]) -> str:
        key_obj = {
            "kind": ticket_payload.get("kind"),
            "messages": ticket_payload.get("messages"),
            "attachments": ticket_payload.get("attachments"),
            "images_count": ticket_payload.get("images_count"),
            "reported_nick": ticket_payload.get("reported_nick"),
            "rule_refs_raw": ticket_payload.get("rule_refs_raw"),
            "server_lang": ticket_payload.get("server_lang"),
            "verdict": ticket_payload.get("verdict"),
            "DUPLICATE_HINT": ticket_payload.get("DUPLICATE_HINT"),
            "IS_SPAM_OR_DUP": ticket_payload.get("IS_SPAM_OR_DUP"),
            "rules": rules_context,
        }
        raw = json.dumps(key_obj, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate_triage(
        self,
        rules_context: str,
        ticket_payload: Dict[str, Any],
        rag_items: List[RuleItem],
    ) -> Tuple[Dict[str, Any], LLMCallMetrics]:
        t0 = time.perf_counter()
        cache_key = self._compute_cache_key(rules_context, ticket_payload)

        if self.cache_enabled and cache_key in self._cache:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            metrics = LLMCallMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                cost_usd=0.0,
                cached=True,
                provider=getattr(self.provider, "model", "argon-4.0"),
            )
            self.call_history.append(metrics)
            return dict(self._cache[cache_key]), metrics

        full_prompt = self.render_prompt(rules_context, ticket_payload)
        last_err: Optional[Exception] = None

        for attempt in range(2):
            try:
                parsed, p_tok, c_tok = self.provider.complete_json(
                    system_prompt=full_prompt,
                    ticket_payload=ticket_payload,
                    rag_items=rag_items,
                    temperature=self.temperature if attempt == 0 else 0.0,
                    max_tokens=self.max_tokens,
                    timeout_sec=self.timeout_sec,
                )
                if not isinstance(parsed, dict) or "category" not in parsed or "draft_reply" not in parsed:
                    raise LLMClientError("Missing required keys in LLM JSON response")

                latency_ms = (time.perf_counter() - t0) * 1000.0
                cost_usd = (p_tok / 1000.0) * self.input_cost_per_1k + (c_tok / 1000.0) * self.output_cost_per_1k
                metrics = LLMCallMetrics(
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=p_tok + c_tok,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    cached=False,
                    provider=getattr(self.provider, "model", "argon-4.0"),
                )
                self.call_history.append(metrics)
                if self.cache_enabled:
                    self._cache[cache_key] = dict(parsed)
                logger.info(
                    "LLM call ok: prompt_tok=%d comp_tok=%d cost=$%.6f latency=%.2fms",
                    p_tok,
                    c_tok,
                    cost_usd,
                    latency_ms,
                )
                return parsed, metrics
            except Exception as exc:
                last_err = exc
                logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)

        raise LLMClientError(f"LLM failed after retry: {last_err}")
