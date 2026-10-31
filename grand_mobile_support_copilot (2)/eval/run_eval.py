#!/usr/bin/env python3
"""
Honest Evaluation Script for Grand Mobile Support Copilot (Case 03, Decentrathon 6.0).
Evaluates on the 1,000-ticket organizer handout valid set (data/tickets_valid.jsonl):
  - Primary mode (--strip-labels / no_verdict_triage_mode): strips verdict, violation,
    punishment, and admin messages BEFORE passing tickets into the pipeline.
  - Also reports full valid record mode (upper bound) and text-only mode (stripping rule_refs_raw).
"""
from collections import Counter, defaultdict
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.fallback import (
    APPEAL_VERDICTS,
    BAN_APPEAL_KEYWORDS,
    BARE_OFFTOPIC_REPORTS,
    CHEAT_WORD_REGEX,
    DM_ABBR_REGEX,
    HACK_KEYWORDS,
    INSULT_WORD_REGEX,
    KB_JB_QUESTION_REGEX,
    KB_NICK_QUESTION_REGEX,
    KB_STUCK_HELP_REGEX,
    KB_TELEPORT_HELP_REGEX,
    KB_VEHICLE_HELP_REGEX,
    PAYMENT_KEYWORDS,
    REPORT_PLAYER_ID_REGEX,
    REPORT_VIOLATION_ABBR_REGEX,
    check_has_evidence,
    compute_priority,
)
from app.langdetect import detect_language
from app.normalize import extract_rule_refs, normalize_text
from app.pipeline import SupportCopilotPipeline
from app.rules_index import RulesIndex
from scripts.predict import strip_post_hoc_labels

CATEGORIES_REQUIRING_RULES = {
    "complaint.insult",
    "complaint.cheat",
    "complaint.dm",
    "complaint.scam",
    "complaint.rmt_ads",
    "complaint.admin",
    "complaint.invalid",
    "report.help",
    "report.account",
    "report.offtopic",
}


def derive_weak_label(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Construct weak ground-truth (category, priority) from the organizer handout record:
    1. First checks admin outcome fields: verdict, violation, punishment, and admin message text.
    2. When verdict is 'other', 'need_evidence', or 'rejected' without an explicit rule in the admin reply,
       uses rule_refs_raw and player text to determine the complaint/report topic.
    """
    kind = str(record.get("kind", "report"))
    violation = str(record.get("violation") or "").lower()
    verdict = str(record.get("verdict") or "").lower()
    raw_refs_str = str(record.get("rule_refs_raw") or "")
    messages = record.get("messages") or []

    player_text = "\n".join(
        str(m.get("text", "")) for m in messages if isinstance(m, dict) and m.get("role", "player") == "player"
    )
    admin_text = "\n".join(
        str(m.get("text", "")) for m in messages if isinstance(m, dict) and m.get("role") == "admin"
    ).lower()

    msg_att = sum(int(m.get("attachments") or 0) for m in messages if isinstance(m, dict))
    total_att = max(int(record.get("attachments") or 0), msg_att)
    norm_text = normalize_text(player_text)
    combined_low = f"{norm_text} {raw_refs_str}".lower()
    refs = set(extract_rule_refs(norm_text, record.get("rule_refs_raw")))
    admin_refs = set(extract_rule_refs(admin_text, violation))
    all_refs = refs | admin_refs
    has_ev = check_has_evidence(total_att, record.get("images_count", 0), norm_text)

    cat: Optional[str] = None
    if kind == "complaint":
        if (
            verdict in APPEAL_VERDICTS
            or "3.4" in all_refs
            or "напишите жалобу на администратора" in admin_text
            or "администратор выдал наказание верно" in admin_text
            or any(k in combined_low for k in BAN_APPEAL_KEYWORDS)
        ):
            cat = "complaint.admin"
        elif "5.2" in admin_refs or "стороннего по" in admin_text or "багоюз" in violation:
            cat = "complaint.cheat"
        elif "4.4" in admin_refs or "2.3" in admin_refs:
            cat = "complaint.rmt_ads"
        elif "4.1" in admin_refs or "4.2" in admin_refs or "обман" in violation:
            cat = "complaint.scam"
        elif "2.1" in admin_refs or "2.4" in admin_refs or "2.6" in admin_refs or "оскорбление" in violation or "упоминание родных" in violation:
            cat = "complaint.insult"
        elif any(r in admin_refs for r in ("5.3", "5.4", "GZ", "GZ 1.1", "GZ 1.2")) or any(w in violation for w in ("dm", "дм", "tk", "sk", "db", "gz", "гз")):
            cat = "complaint.dm"
        elif verdict == "invalid_format" or "ЖБ 1.1.2" in all_refs or "ЖБ 1.2.7" in all_refs or "1.1.2 правил подачи" in admin_text or "1.2.7 правил подачи" in admin_text:
            cat = "complaint.invalid"
        elif "5.2" in refs or bool(CHEAT_WORD_REGEX.search(combined_low)):
            cat = "complaint.cheat"
        elif "2.3" in refs or "4.4" in refs or any(k in combined_low for k in ("продаёт вирты", "продает вирты", "за реал", "реальные деньги", "рекламирует", "rmt")):
            cat = "complaint.rmt_ads"
        elif "4.1" in refs or "4.2" in refs or any(k in combined_low for k in ("обманул", "обман", "скам", "кинул на", "при обмене", "не отдал", "вымогает", "алдап", "aldab", "abmen", "scam")):
            cat = "complaint.scam"
        elif "2.1" in refs or "2.4" in refs or "2.6" in refs or bool(INSULT_WORD_REGEX.search(combined_low)):
            cat = "complaint.insult"
        elif any(r in refs for r in ("5.3", "5.4", "GZ", "GZ 1.1", "GZ 1.2")) or bool(DM_ABBR_REGEX.search(combined_low)) or any(
            k in combined_low for k in ("убил", "стрелял", "стрельб", "зелёной зоне", "зеленой зоне", "задавил", "oldirdi", "uldirdi", "uldrdi", "uryapdi", "связал низа что", "наручники", "kishanlab", "qamoqga")
        ):
            cat = "complaint.dm"
        elif detect_language(combined_low) == "und" or len(norm_text.strip()) < 3:
            cat = "complaint.invalid"
        else:
            cat = "complaint.other"
    else:
        text_low = norm_text.lower()
        clean_punct = re.sub(r"[!?.…,\s]+", " ", text_low).strip()
        has_rule_ref_in_report = bool(refs) or bool(re.search(r"\b\d+\.\d+\b", text_low))
        has_violation_abbr = bool(REPORT_VIOLATION_ABBR_REGEX.search(text_low))
        has_player_id_num = bool(REPORT_PLAYER_ID_REGEX.search(text_low)) and "?" not in text_low

        if "1.4" in all_refs or any(k in text_low for k in HACK_KEYWORDS):
            cat = "report.account"
        elif any(k in text_low for k in PAYMENT_KEYWORDS):
            cat = "report.payment"
        elif KB_NICK_QUESTION_REGEX.search(text_low) or KB_JB_QUESTION_REGEX.search(text_low):
            cat = "report.question"
        elif KB_VEHICLE_HELP_REGEX.search(text_low) or KB_STUCK_HELP_REGEX.search(text_low) or KB_TELEPORT_HELP_REGEX.search(text_low):
            cat = "report.help"
        elif clean_punct in BARE_OFFTOPIC_REPORTS:
            cat = "report.offtopic"
        elif (
            "слежу за игроком" in admin_text
            or has_rule_ref_in_report
            or has_violation_abbr
            or has_player_id_num
            or any(k in text_low for k in ("тут дм", "меня дм", "связали", "зав'язали", "заломал", "у него что-то есть", "пристаёт", "пристаьот", "меня бьют", "убивают"))
        ):
            cat = "report.complaint"
        elif "приятной игры" in admin_text or detect_language(norm_text) == "und":
            cat = "report.offtopic"
        elif "изучите правила" in admin_text or "не владеем данной информацией" in admin_text or "?" in norm_text or any(
            k in text_low for k in ("как ", "что будет", "что такое", "где ", "когда ", "будет сегодня")
        ):
            cat = "report.question"
        elif len(clean_punct.split()) <= 2:
            cat = "report.offtopic"
        else:
            cat = "report.question"

    prio = compute_priority(cat, has_ev, norm_text) if cat else None
    return cat, prio


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    f1s: List[float] = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def build_stratified_100_subset(
    records: List[Dict[str, Any]],
    output_path: str = "data/manual_annotation_100.jsonl",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_reporter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        rep = str(r.get("reporter_hash") or r.get("ticket_id"))
        by_reporter[rep].append(r)

    lang_groups: Dict[str, List[str]] = defaultdict(list)
    for rep, items in by_reporter.items():
        lg = items[0].get("lang") or detect_language(normalize_text((items[0].get("messages") or [{}])[0].get("text", "")))
        lang_groups[str(lg)].append(rep)

    selected_tickets: List[Dict[str, Any]] = []
    selected_reporters = set()

    lang_order = ["kk", "uz", "ru", "lat", "und", "pt", "en", "es"]
    active = True
    while len(selected_tickets) < 100 and active:
        active = False
        for lg in lang_order:
            reps = lang_groups.get(lg, [])
            while reps:
                rep = reps.pop(0)
                if rep in selected_reporters:
                    continue
                group_items = by_reporter[rep]
                if len(selected_tickets) + len(group_items) <= 100:
                    selected_reporters.add(rep)
                    selected_tickets.extend(group_items)
                    active = True
                    break

    selected_tickets = selected_tickets[:100]
    holdout_reporters = set(by_reporter.keys()) - selected_reporters
    subset_lang_counts: Counter = Counter()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in selected_tickets:
            weak_cat, weak_prio = derive_weak_label(r)
            msgs = r.get("messages") or []
            txt = msgs[0].get("text", "") if msgs else ""
            dataset_lang = str(r.get("lang") or detect_language(normalize_text(txt)))
            subset_lang_counts[dataset_lang] += 1
            row = {
                "ticket_id": r.get("ticket_id"),
                "reporter_hash": r.get("reporter_hash"),
                "kind": r.get("kind"),
                "lang": dataset_lang,
                "weak_category": weak_cat,
                "weak_priority": weak_prio,
                "manual_category": weak_cat,
                "manual_priority": weak_prio,
                "annotator_notes": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_info = {
        "subset_size": len(selected_tickets),
        "subset_unique_reporters": len(selected_reporters),
        "holdout_unique_reporters": len(holdout_reporters),
        "reporter_overlap_count": len(selected_reporters & holdout_reporters),
        "subset_lang_distribution": dict(subset_lang_counts),
    }
    return selected_tickets, split_info


def _strip_post_hoc_fields(rec: Dict[str, Any], strip_rule_refs_raw: bool = False) -> Dict[str, Any]:
    clean = strip_post_hoc_labels(rec)
    if strip_rule_refs_raw:
        clean.pop("rule_refs_raw", None)
    return clean


def run_evaluation(valid_path: str = "data/tickets_valid.jsonl") -> Dict[str, Any]:
    with open(valid_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    _, split_info = build_stratified_100_subset(records)
    rules_index = RulesIndex()
    pipeline = SupportCopilotPipeline()

    y_cat_true: List[str] = []
    y_prio_true: List[str] = []

    # Primary Mode (--strip-labels / no_verdict_triage_mode)
    pipeline.deduplicator.reset()
    y_cat_pred_no_verdict: List[str] = []
    y_prio_pred_no_verdict: List[str] = []
    needs_human_flags: List[bool] = []
    empty_cited_all_flags: List[bool] = []
    empty_cited_required_flags: List[bool] = []
    hallucinated_cited_flags: List[bool] = []
    confidences: List[float] = []
    mismatch_counter_no_verdict: Counter = Counter()

    by_lang_cat: Dict[str, List[bool]] = defaultdict(list)
    by_lang_prio_true: Dict[str, List[str]] = defaultdict(list)
    by_lang_prio_pred: Dict[str, List[str]] = defaultdict(list)
    by_lang_reply_match: Dict[str, List[bool]] = defaultdict(list)

    for rec in records:
        weak_cat, weak_prio = derive_weak_label(rec)
        stripped = _strip_post_hoc_fields(rec, strip_rule_refs_raw=False)
        pred = pipeline.process_ticket(stripped)
        ds_lang = str(rec.get("lang") or pred.lang)

        if weak_cat and weak_prio:
            y_cat_true.append(weak_cat)
            y_prio_true.append(weak_prio)
            y_cat_pred_no_verdict.append(pred.category)
            y_prio_pred_no_verdict.append(pred.priority)
            if weak_cat != pred.category:
                mismatch_counter_no_verdict[f"{weak_cat} -> {pred.category}"] += 1

            by_lang_cat[ds_lang].append(pred.category == weak_cat)
            by_lang_prio_true[ds_lang].append(weak_prio)
            by_lang_prio_pred[ds_lang].append(pred.priority)

        needs_human_flags.append(pred.needs_human)
        confidences.append(pred.confidence)
        is_empty_cited = len(pred.draft_reply.cited) == 0
        empty_cited_all_flags.append(is_empty_cited)
        if pred.category in CATEGORIES_REQUIRING_RULES:
            empty_cited_required_flags.append(is_empty_cited)

        has_hallucinated = any(
            (not rules_index.exists(cid)) or (cid not in pred.draft_reply.text)
            for cid in pred.draft_reply.cited
        )
        hallucinated_cited_flags.append(has_hallucinated)

        detected_draft_lang = detect_language(pred.draft_reply.text)
        by_lang_reply_match[ds_lang].append(
            pred.draft_reply.lang not in ("lat", "und")
            and (pred.lang in ("lat", "und") or detected_draft_lang == pred.draft_reply.lang)
        )

    # Full record mode (upper bound with verdict)
    pipeline.deduplicator.reset()
    y_cat_pred_full: List[str] = []
    y_prio_pred_full: List[str] = []
    for rec in records:
        pred_full = pipeline.process_ticket(rec)
        y_cat_pred_full.append(pred_full.category)
        y_prio_pred_full.append(pred_full.priority)

    # Text-only mode (also stripping rule_refs_raw)
    pipeline.deduplicator.reset()
    y_cat_pred_text_only: List[str] = []
    y_prio_pred_text_only: List[str] = []
    for rec in records:
        stripped_text = _strip_post_hoc_fields(rec, strip_rule_refs_raw=True)
        pred3 = pipeline.process_ticket(stripped_text)
        y_cat_pred_text_only.append(pred3.category)
        y_prio_pred_text_only.append(pred3.priority)

    total = len(records)
    cat_acc = sum(1 for t, p in zip(y_cat_true, y_cat_pred_full) if t == p) / float(len(y_cat_true))
    prio_f1 = macro_f1(y_prio_true, y_prio_pred_full, ["p0", "p1", "p2", "p3"])

    cat_acc_no_verdict = sum(1 for t, p in zip(y_cat_true, y_cat_pred_no_verdict) if t == p) / float(len(y_cat_true))
    prio_f1_no_verdict = macro_f1(y_prio_true, y_prio_pred_no_verdict, ["p0", "p1", "p2", "p3"])

    cat_acc_text_only = sum(1 for t, p in zip(y_cat_true, y_cat_pred_text_only) if t == p) / float(len(y_cat_true))
    prio_f1_text_only = macro_f1(y_prio_true, y_prio_pred_text_only, ["p0", "p1", "p2", "p3"])

    kk_acc = (sum(by_lang_cat["kk"]) / len(by_lang_cat["kk"])) if by_lang_cat["kk"] else 0.0
    kk_f1 = macro_f1(by_lang_prio_true["kk"], by_lang_prio_pred["kk"], ["p0", "p1", "p2", "p3"])
    uz_acc = (sum(by_lang_cat["uz"]) / len(by_lang_cat["uz"])) if by_lang_cat["uz"] else 0.0
    uz_f1 = macro_f1(by_lang_prio_true["uz"], by_lang_prio_pred["uz"], ["p0", "p1", "p2", "p3"])

    nh_count = sum(needs_human_flags)
    auto_closed_count = total - nh_count

    metrics = {
        "total_valid_tickets": total,
        "complaints_count": sum(1 for r in records if r.get("kind") == "complaint"),
        "reports_count": sum(1 for r in records if r.get("kind") == "report"),
        "primary_mode": "strip_labels (verdict, violation, punishment, and role=admin removed before pipeline)",
        "reporter_hash_split": split_info,
        "category_accuracy_no_verdict_triage_mode": round(cat_acc_no_verdict, 4),
        "priority_macro_f1_no_verdict_triage_mode": round(prio_f1_no_verdict, 4),
        "category_accuracy_full": round(cat_acc, 4),
        "priority_macro_f1_full": round(prio_f1, 4),
        "category_accuracy_text_only_no_rule_refs_raw": round(cat_acc_text_only, 4),
        "priority_macro_f1_text_only_no_rule_refs_raw": round(prio_f1_text_only, 4),
        "needs_human_count": nh_count,
        "needs_human_rate": round(nh_count / float(total), 4),
        "auto_closed_count": auto_closed_count,
        "auto_closed_rate": round(auto_closed_count / float(total), 4),
        "empty_cited_overall_rate": round(sum(empty_cited_all_flags) / float(total), 4),
        "empty_cited_when_required_rate": round(sum(empty_cited_required_flags) / float(max(1, len(empty_cited_required_flags))), 4),
        "hallucinated_cited_rate": round(sum(hallucinated_cited_flags) / float(total), 4),
        "confidence_stats": {
            "min": min(confidences),
            "max": max(confidences),
            "avg": round(sum(confidences) / float(len(confidences)), 4),
            "distinct_values": sorted(list(set(confidences))),
        },
        "kk_metrics": {
            "count": len(by_lang_cat["kk"]),
            "category_accuracy": round(kk_acc, 4),
            "priority_macro_f1": round(kk_f1, 4),
            "reply_lang_valid_rate": round(sum(by_lang_reply_match["kk"]) / max(1, len(by_lang_reply_match["kk"])), 4),
        },
        "uz_metrics": {
            "count": len(by_lang_cat["uz"]),
            "category_accuracy": round(uz_acc, 4),
            "priority_macro_f1": round(uz_f1, 4),
            "reply_lang_valid_rate": round(sum(by_lang_reply_match["uz"]) / max(1, len(by_lang_reply_match["uz"])), 4),
        },
        "top_category_mismatches": dict(mismatch_counter_no_verdict.most_common(10)),
    }

    os.makedirs("data", exist_ok=True)
    with open("data/eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    run_evaluation()
