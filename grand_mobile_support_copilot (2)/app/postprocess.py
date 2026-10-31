import re
from typing import Any, Dict, List, Optional, Set
from app.fallback import (
    APPEAL_VERDICTS,
    build_safe_draft_text,
    check_has_evidence,
    compute_needs_human,
    compute_priority,
    extract_accused_nickname,
    extract_server_from_text,
    infer_category,
    select_best_citations,
)
from app.langdetect import detect_language, resolve_reply_lang
from app.normalize import extract_rule_refs, fix_cp1251_kazakh, normalize_text
from app.rules_index import RuleItem, RulesIndex
from app.schemas import ALLOWED_CATEGORIES, DraftReply, ExtractedData, TriageResponse

FORBIDDEN_DRAFT_PATTERNS = [
    # Russian
    re.compile(r"\b(?:мы\s+вас\s+разблокируем|мы\s+вас\s+разбаним|разбаним\s+вас|бан\s+будет\s+снят|аккаунт\s+будет\s+восстановлен|восстановим\s+аккаунт)\b", re.IGNORECASE),
    re.compile(r"\b(?:вам\s+вернут|вернём\s+деньги|вернем\s+деньги|вернём\s+средства|компенсируем|выплатим\s+компенсацию|возврат\s+будет\s+оформлен)\b", re.IGNORECASE),
    re.compile(r"\b(?:игрок\s+\S+\s+получит|игрок\s+получит|нарушитель\s+получит|получит\s+мут|получит\s+бан|будет\s+забанен|будет\s+наказан|мы\s+забаним|мы\s+накажем)\b", re.IGNORECASE),
    re.compile(r"\b(?:нарушение\s+подтверждено|по\s+скриншотам?\s+видно|на\s+скриншоте\s+видно|на\s+видео\s+видно)\b", re.IGNORECASE),
    re.compile(r"\b(?:бан|мут|варн|блокировк[ау])\s+на\s+\d+\s*(?:дн|сут|час|мин)", re.IGNORECASE),
    re.compile(r"(?:спешит\s+к\s+вам\s+на\s+помощь|уже\s+спешит)", re.IGNORECASE),
    # Kazakh
    re.compile(r"(?:бұғаттан\s+шығарамыз|баннан\s+шығарамыз|аккаунтыңыз\s+қалпына\s+келтіріледі|ақшаңызды\s+қайтарамыз|өтемақы\s+төлейміз)", re.IGNORECASE),
    re.compile(r"(?:ойыншы\s+бан\s+алады|ойыншы\s+мут\s+алады|міндетті\s+түрде\s+жазаланады|бұзушылық\s+расталды|скриншоттан\s+көрініп\s+тұр|көмекке\s+келе\s+жатыр)", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:күнге|сағатқа|минутқа)\s+(?:бан|мут)\b", re.IGNORECASE),
    # Uzbek
    re.compile(r"(?:bandan\s+chiqaramiz|akkauntingiz\s+tiklanadi|pulingizni\s+qaytaramiz|kompensatsiya\s+beramiz)", re.IGNORECASE),
    re.compile(r"(?:o'yinchi\s+ban\s+oladi|o'yinchi\s+mut\s+oladi|jazolanadi|qoidabuzarlik\s+tasdiqlandi|skrinshotda\s+ko'rinib\s+turibdi|yordamga\s+kelmoqda)", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:kunga|soatga|daqiqaga)\s+(?:ban|mut)\b", re.IGNORECASE),
    # English / Spanish / Portuguese
    re.compile(r"\b(?:will\s+be\s+banned|will\s+get\s+muted|will\s+be\s+punished|we\s+will\s+unban\s+you|we\s+will\s+refund|violation\s+is\s+confirmed|visible\s+in\s+the\s+screenshots?|on\s+the\s+way\s+to\s+assist)\b", re.IGNORECASE),
    re.compile(r"\b(?:banned\s+for\s+\d+\s*(?:days?|hours?)|muted\s+for\s+\d+\s*(?:minutes?|hours?))\b", re.IGNORECASE),
]

# Matches rule numbers in draft text such as "2.1", "1.1.2", "ЖБ 1.1.2", "GZ 1.1"
DRAFT_RULE_NUMBER_REGEX = re.compile(
    r"(?:(?:ЖБ|GZ)\s+)?\b\d+\.\d+(?:\.\d+)?\b",
    re.UNICODE,
)


def contains_forbidden_phrases(text: str) -> bool:
    if not text:
        return True
    for pat in FORBIDDEN_DRAFT_PATTERNS:
        if pat.search(text):
            return True
    return False


def extract_rule_numbers_from_draft_text(draft_text: str) -> List[str]:
    """Extract all rule number tokens (e.g. '2.1', 'ЖБ 1.1.2', 'GZ 1.1') appearing in draft_reply.text."""
    if not draft_text:
        return []
    found: List[str] = []
    for m in DRAFT_RULE_NUMBER_REGEX.finditer(draft_text):
        token = re.sub(r"\s+", " ", m.group(0).strip())
        if token not in found:
            found.append(token)
    return found


def has_citation_text_mismatch(draft_text: str, cited: List[str]) -> bool:
    """
    Task 3 deterministic citation guard:
    1) Every citation in `cited` must literally appear in `draft_reply.text`.
    2) `draft_reply.text` must NOT contain any rule numbers (e.g. '9.9', '3.1') that are not in `cited`.
    """
    if not draft_text:
        return True
    cited_set = {str(c).strip() for c in (cited or []) if c}
    for cid in cited_set:
        if cid not in draft_text:
            return True
    text_rule_nums = extract_rule_numbers_from_draft_text(draft_text)
    for num in text_rule_nums:
        if num not in cited_set:
            return True
    return False


def enforce_kind_category_alignment(
    kind: str,
    category: str,
    text: str,
    rule_refs: List[str],
    has_evidence: bool,
    is_spam: bool,
    verdict: Optional[str] = None,
) -> str:
    expected_prefix = "complaint." if kind == "complaint" else "report."
    if (verdict or "").strip().lower() in APPEAL_VERDICTS and kind == "complaint":
        return "complaint.admin"
    if category in ALLOWED_CATEGORIES and category.startswith(expected_prefix):
        return category

    cross_map = {
        ("complaint", "report.complaint"): "complaint.other",
        ("complaint", "report.offtopic"): "complaint.invalid",
        ("complaint", "report.question"): "complaint.invalid",
        ("report", "complaint.insult"): "report.complaint",
        ("report", "complaint.dm"): "report.complaint",
        ("report", "complaint.cheat"): "report.complaint",
        ("report", "complaint.scam"): "report.complaint",
        ("report", "complaint.invalid"): "report.offtopic",
    }
    if (kind, category) in cross_map:
        mapped = cross_map[(kind, category)]
        if mapped == "complaint.other":
            return infer_category(text, kind, rule_refs, has_evidence, is_spam=is_spam, verdict=verdict)
        return mapped

    return infer_category(text, kind, rule_refs, has_evidence, is_spam=is_spam, verdict=verdict)


def filter_valid_cited(
    raw_cited: Any,
    draft_text: str,
    rag_rule_ids: Set[str],
    rules_index: RulesIndex,
) -> List[str]:
    if not isinstance(raw_cited, list) or not draft_text:
        return []
    valid: List[str] = []
    seen: Set[str] = set()
    for item in raw_cited:
        if not item or not isinstance(item, str):
            continue
        cid = item.strip()
        if cid in seen:
            continue
        if not rules_index.exists(cid):
            continue
        if cid not in rag_rule_ids:
            continue
        if cid not in draft_text:
            continue
        seen.add(cid)
        valid.append(cid)
    return valid


def postprocess_triage_dict(
    raw_data: Dict[str, Any],
    player_text: str,
    kind: str,
    server_lang: Optional[str],
    region: Optional[str],
    reported_nick: Optional[str],
    rule_refs_raw: Optional[Any],
    attachments: Optional[int],
    images_count: Optional[int],
    is_spam_or_dup_detected: bool,
    rag_items: List[RuleItem],
    rules_index: RulesIndex,
    verdict: Optional[str] = None,
) -> TriageResponse:
    norm_text = normalize_text(player_text)
    rag_rule_ids = {it.id for it in rag_items}

    detected_lang = detect_language(norm_text)
    raw_lang = str(raw_data.get("lang", "")).strip().lower()
    valid_langs = {"ru", "kk", "uz", "en", "es", "pt", "lat", "und"}
    if detected_lang in ("kk", "uz", "und", "lat") or raw_lang not in valid_langs:
        final_lang = detected_lang
    else:
        final_lang = raw_lang

    true_has_evidence = check_has_evidence(attachments, images_count, norm_text)
    true_rule_refs = extract_rule_refs(norm_text, rule_refs_raw)
    raw_extracted = raw_data.get("extracted") if isinstance(raw_data.get("extracted"), dict) else {}

    llm_refs_raw = raw_extracted.get("rule_refs", [])
    merged_refs = list(true_rule_refs)
    for r in extract_rule_refs(None, llm_refs_raw):
        if r not in merged_refs:
            merged_refs.append(r)

    # extracted.server — только явно названный в тексте сервер (§8.1).
    # Значения region/server_lang и любые догадки провайдера сюда не подставляются.
    final_server = extract_server_from_text(norm_text)

    true_nick = extract_accused_nickname(norm_text, reported_nick)
    raw_nick = raw_extracted.get("nickname")
    if true_nick:
        final_nick = true_nick
    elif isinstance(raw_nick, str) and raw_nick.strip() and raw_nick.strip() in norm_text:
        final_nick = raw_nick.strip()
    else:
        final_nick = None

    raw_spam = bool(raw_data.get("is_spam_or_duplicate", False))
    final_spam = bool(is_spam_or_dup_detected or raw_spam or (str(verdict or "").lower() == "duplicate"))

    raw_category = str(raw_data.get("category", "")).strip()
    final_category = enforce_kind_category_alignment(
        kind=kind,
        category=raw_category,
        text=norm_text,
        rule_refs=merged_refs,
        has_evidence=true_has_evidence,
        is_spam=final_spam,
        verdict=verdict,
    )

    final_priority = compute_priority(final_category, true_has_evidence, norm_text)

    try:
        raw_conf = float(raw_data.get("confidence", 0.8))
    except (ValueError, TypeError):
        raw_conf = 0.8
    final_confidence = max(0.0, min(1.0, round(raw_conf, 4)))

    raw_draft = raw_data.get("draft_reply") if isinstance(raw_data.get("draft_reply"), dict) else {}
    raw_draft_lang = str(raw_draft.get("lang", "")).strip().lower()
    expected_reply_lang = resolve_reply_lang(final_lang, server_lang)
    valid_reply_langs = {"ru", "kk", "uz", "en", "es", "pt"}

    if raw_draft_lang not in valid_reply_langs or raw_draft_lang != expected_reply_lang:
        final_reply_lang = expected_reply_lang
    else:
        final_reply_lang = raw_draft_lang

    raw_draft_text = fix_cp1251_kazakh(str(raw_draft.get("text", "")).strip())
    draft_text_lang = detect_language(raw_draft_text) if raw_draft_text else "und"

    lang_mismatch = (
        draft_text_lang in ("lat", "und")
        or (final_reply_lang in ("ru", "kk", "uz", "en") and draft_text_lang != final_reply_lang)
    )
    best_cids = select_best_citations(final_category, merged_refs, rag_items, true_has_evidence, text=norm_text, verdict=verdict)
    allowed_cids_set = set(best_cids)

    if not raw_draft_text or contains_forbidden_phrases(raw_draft_text) or lang_mismatch:
        final_draft_text, final_cited = build_safe_draft_text(
            category=final_category,
            reply_lang=final_reply_lang,
            nickname=final_nick,
            cited_id=best_cids,
            has_evidence=true_has_evidence,
            rag_items=rag_items,
            verdict=verdict,
            player_text=norm_text,
            player_rule_refs=merged_refs,
            server=final_server,
        )
    else:
        final_draft_text = raw_draft_text
        candidate_cited = filter_valid_cited(
            raw_cited=raw_draft.get("cited", []),
            draft_text=final_draft_text,
            rag_rule_ids=rag_rule_ids,
            rules_index=rules_index,
        )
        # Keep only citations that are relevant to the category/player refs
        final_cited = [c for c in candidate_cited if c in allowed_cids_set]

    final_cited = filter_valid_cited(
        raw_cited=final_cited,
        draft_text=final_draft_text,
        rag_rule_ids=rag_rule_ids,
        rules_index=rules_index,
    )

    # Task 3 guard: if draft_reply.text contains any rule numbers not in final_cited
    # or any citation from final_cited is missing in draft_reply.text, rebuild safe draft text
    if has_citation_text_mismatch(final_draft_text, final_cited):
        valid_best_cids = [c for c in best_cids if c in rag_rule_ids and rules_index.exists(c)]
        final_draft_text, final_cited = build_safe_draft_text(
            category=final_category,
            reply_lang=final_reply_lang,
            nickname=final_nick,
            cited_id=valid_best_cids,
            has_evidence=true_has_evidence,
            rag_items=rag_items,
            verdict=verdict,
            player_text=norm_text,
            player_rule_refs=merged_refs,
            server=final_server,
        )
        final_cited = filter_valid_cited(
            raw_cited=final_cited,
            draft_text=final_draft_text,
            rag_rule_ids=rag_rule_ids,
            rules_index=rules_index,
        )

    rules_degraded = bool(rules_index.load_error or len(rules_index.index) == 0)
    rag_has_answer = bool(final_cited) and not rules_degraded
    raw_needs_human = bool(raw_data.get("needs_human", True))
    required_needs_human = compute_needs_human(
        category=final_category,
        text=norm_text,
        confidence=final_confidence,
        rag_has_answer=rag_has_answer,
        verdict=verdict,
    )
    final_needs_human = True if (required_needs_human or rules_degraded) else raw_needs_human

    raw_reasons = raw_data.get("reasons", [])
    if isinstance(raw_reasons, list):
        clean_reasons = [str(r).strip() for r in raw_reasons if str(r).strip()][:3]
    else:
        clean_reasons = []
    if not clean_reasons:
        clean_reasons = [
            f"обращение классифицировано как {final_category}",
            "требуется проверка администратором" if final_needs_human else "содержательный ответ подтверждён базой правил",
        ]
    if rules_degraded:
        final_needs_human = True
        if "rules.json не загружен" not in clean_reasons:
            clean_reasons = list(clean_reasons[:2]) + ["rules.json не загружен"]

    return TriageResponse(
        lang=final_lang,  # type: ignore[arg-type]
        category=final_category,  # type: ignore[arg-type]
        priority=final_priority,  # type: ignore[arg-type]
        is_spam_or_duplicate=final_spam,
        extracted=ExtractedData(
            nickname=final_nick,
            rule_refs=merged_refs,
            has_evidence=true_has_evidence,
            server=final_server,
        ),
        draft_reply=DraftReply(
            lang=final_reply_lang,  # type: ignore[arg-type]
            text=final_draft_text,
            cited=final_cited,
        ),
        needs_human=final_needs_human,
        confidence=final_confidence,
        reasons=clean_reasons,
    )
