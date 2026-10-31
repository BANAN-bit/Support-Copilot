#!/usr/bin/env python3
"""
Citation Verification Script (eval/check_citations.py).
Verifies citations across all 1,000 tickets of data/tickets_valid.jsonl (in --strip-labels mode):
1. Literal existence check directly against the raw JSON literals in data/rules.json and data/taxonomy.json:
   - Section 30 ("Общие правила игры"): bare refs from points[].ref ("1.1".."5.9")
   - Section 31 ("Правила жалоб"): "ЖБ " + points[].ref ("ЖБ 1.1".."ЖБ 1.5.2", confirmed by taxonomy.json)
   - Section 32 ("Правила нахождения в Зелёной Зоне"): "GZ " + points[].ref ("GZ 1.1".."GZ 2.3") + "GZ" (from taxonomy.json)
2. Bidirectional consistency check between `draft_reply.cited` and `draft_reply.text`:
   - Every citation in `cited` appears in `draft_reply.text` (`missing_in_draft_text == 0`)
   - No rule numbers appear in `draft_reply.text` that are not in `cited` (`extra_rule_numbers_in_draft_text == 0`)
3. Relevance check:
   Every citation must either belong to `rules` of the predicted category from `data/taxonomy.json`
   OR be explicitly specified by the player in `text` or `rule_refs_raw`.
   Outputs the count and detailed examples of any relevance violations.
"""
from collections import Counter
import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.normalize import extract_rule_refs, normalize_text
from app.pipeline import SupportCopilotPipeline
from app.postprocess import extract_rule_numbers_from_draft_text
from scripts.predict import strip_post_hoc_labels


def load_raw_allowed_citations_and_taxonomy_map(
    rules_path: str = "data/rules.json",
    taxonomy_path: str = "data/taxonomy.json",
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    with open(rules_path, "r", encoding="utf-8") as f:
        rules_data = json.load(f)
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        tax_data = json.load(f)

    allowed: Set[str] = set()
    for sec in rules_data.get("rules", []):
        sid = sec.get("id")
        pts = sec.get("points", [])
        if sid == 30:
            for p in pts:
                allowed.add(str(p["ref"]).strip())
        elif sid == 31:
            for p in pts:
                allowed.add(f"ЖБ {str(p['ref']).strip()}")
        elif sid == 32:
            for p in pts:
                allowed.add(f"GZ {str(p['ref']).strip()}")

    taxonomy_cat_rules: Dict[str, Set[str]] = {}
    for cat in tax_data.get("categories", []):
        code = str(cat.get("code", "")).strip()
        cat_rules_set: Set[str] = set()
        for r in cat.get("rules", []):
            r_str = str(r).strip()
            allowed.add(r_str)
            cat_rules_set.add(r_str)
        if code:
            taxonomy_cat_rules[code] = cat_rules_set

    return allowed, taxonomy_cat_rules


def _classify_prefix(cid: str) -> str:
    if cid.startswith("ЖБ "):
        return "ЖБ"
    if cid == "GZ" or cid.startswith("GZ "):
        return "GZ"
    if cid and cid[0].isdigit():
        return "bare_number (ОП)"
    return "other"


def _extract_player_text(clean_rec: Dict[str, Any]) -> str:
    msgs = clean_rec.get("messages") or []
    parts = [
        str(m.get("text", ""))
        for m in msgs
        if isinstance(m, dict) and m.get("role", "player") == "player"
    ]
    if parts:
        return "\n".join(p for p in parts if p).strip()
    return str(clean_rec.get("text") or "").strip()


def run_check_citations(
    valid_path: str = "data/tickets_valid.jsonl",
    strip_labels: bool = True,
) -> Dict[str, Any]:
    with open(valid_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    allowed_literals, taxonomy_cat_rules = load_raw_allowed_citations_and_taxonomy_map()
    pipeline = SupportCopilotPipeline(rules_path="data/rules.json")
    pipeline.deduplicator.reset()

    total_tickets = len(records)
    total_cited_items = 0
    missing_in_rules_literals = 0
    missing_in_draft_text = 0
    extra_rule_numbers_in_draft_text = 0
    tickets_with_hallucinations = 0
    tickets_with_empty_cited = 0
    citations_by_prefix: Counter = Counter()
    citations_by_exact_id: Counter = Counter()
    empty_cited_by_category: Counter = Counter()
    non_empty_cited_by_category: Counter = Counter()

    relevance_violations_count = 0
    tickets_with_relevance_violations = 0
    relevance_violation_examples: List[Dict[str, Any]] = []

    for rec in records:
        clean_rec = strip_post_hoc_labels(rec) if strip_labels else rec
        resp = pipeline.process_ticket(clean_rec)
        cited = resp.draft_reply.cited
        draft_text = resp.draft_reply.text

        player_text = _extract_player_text(clean_rec)
        norm_text = normalize_text(player_text)
        explicit_player_refs = set(extract_rule_refs(norm_text, clean_rec.get("rule_refs_raw")))
        allowed_for_cat = taxonomy_cat_rules.get(resp.category, set())

        if not cited:
            tickets_with_empty_cited += 1
            empty_cited_by_category[resp.category] += 1
        else:
            non_empty_cited_by_category[resp.category] += 1

        ticket_bad = False
        ticket_irrelevant = False
        cited_set = set(cited)
        for cid in cited:
            total_cited_items += 1
            citations_by_prefix[_classify_prefix(cid)] += 1
            citations_by_exact_id[cid] += 1
            if cid not in allowed_literals:
                missing_in_rules_literals += 1
                ticket_bad = True
            if cid not in draft_text:
                missing_in_draft_text += 1
                ticket_bad = True
            if cid not in allowed_for_cat and cid not in explicit_player_refs:
                relevance_violations_count += 1
                ticket_irrelevant = True
                relevance_violation_examples.append(
                    {
                        "ticket_id": rec.get("ticket_id"),
                        "category": resp.category,
                        "cited": cid,
                        "taxonomy_category_rules": sorted(allowed_for_cat),
                        "explicit_player_refs": sorted(explicit_player_refs),
                        "rule_refs_raw": clean_rec.get("rule_refs_raw"),
                        "text": player_text[:120],
                    }
                )
        for text_ref in extract_rule_numbers_from_draft_text(draft_text):
            if text_ref not in cited_set:
                extra_rule_numbers_in_draft_text += 1
                ticket_bad = True
        if ticket_bad:
            tickets_with_hallucinations += 1
        if ticket_irrelevant:
            tickets_with_relevance_violations += 1

    report = {
        "total_tickets_checked": total_tickets,
        "strip_labels": strip_labels,
        "allowed_literals_from_rules_and_taxonomy": len(allowed_literals),
        "total_cited_items_checked": total_cited_items,
        "missing_in_rules_literals": missing_in_rules_literals,
        "missing_in_draft_text": missing_in_draft_text,
        "extra_rule_numbers_in_draft_text": extra_rule_numbers_in_draft_text,
        "tickets_with_hallucinations": tickets_with_hallucinations,
        "hallucination_rate": round(tickets_with_hallucinations / float(max(1, total_tickets)), 6),
        "relevance_violations_count": relevance_violations_count,
        "tickets_with_relevance_violations": tickets_with_relevance_violations,
        "relevance_violation_rate": round(
            relevance_violations_count / float(max(1, total_cited_items)), 6
        ),
        "relevance_violation_examples": relevance_violation_examples,
        "citations_by_prefix": dict(citations_by_prefix),
        "citations_by_exact_id": dict(citations_by_exact_id.most_common()),
        "tickets_with_empty_cited": tickets_with_empty_cited,
        "empty_cited_rate": round(tickets_with_empty_cited / float(max(1, total_tickets)), 4),
        "empty_cited_by_category": dict(empty_cited_by_category),
        "non_empty_cited_by_category": dict(non_empty_cited_by_category),
    }

    os.makedirs("data", exist_ok=True)
    with open("data/check_citations.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if tickets_with_hallucinations > 0:
        sys.exit(1)
    return report


if __name__ == "__main__":
    run_check_citations()
