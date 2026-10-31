#!/usr/bin/env python3
"""
Batch JSONL triage processor for Grand Mobile Support Copilot.

Usage:
    python3 scripts/predict.py --input tickets.jsonl --output pred.jsonl [--workers 8] [--mode MODE]

Режимы (--mode):
    full          — запись передаётся в пайплайн как есть (верхняя оценка с полем verdict);
    strip-labels  — удаляются послефактум-поля администратора (verdict, violation, punishment,
                    category, priority) и сообщения role="admin" — режим по умолчанию, он же
                    используется в POST /triage и в eval/run_eval.py;
    text-only     — strip-labels + удаление rule_refs_raw (только сырой текст обращения).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.dedup import Deduplicator
from app.fallback import build_fallback_response
from app.pipeline import SupportCopilotPipeline
from app.schemas import TicketRequest


PREDICT_MODES: Tuple[str, ...] = ("full", "strip-labels", "text-only")


def strip_post_hoc_labels(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Point 3: Remove post-hoc admin fields (verdict, violation, punishment) and
    admin messages (role=='admin') BEFORE passing the ticket into the pipeline.
    """
    clean = dict(raw_obj)
    clean.pop("verdict", None)
    clean.pop("violation", None)
    clean.pop("punishment", None)
    clean.pop("category", None)
    clean.pop("priority", None)
    msgs = clean.get("messages")
    if isinstance(msgs, list):
        clean["messages"] = [
            m for m in msgs if isinstance(m, dict) and m.get("role") != "admin"
        ]
    return clean


def apply_predict_mode(raw_obj: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Применяет режим прогона к исходной записи тикета до передачи в пайплайн."""
    if mode not in PREDICT_MODES:
        raise ValueError(f"Unknown predict mode: {mode!r}; expected one of {PREDICT_MODES}")
    clean = dict(raw_obj)
    if mode == "full":
        return clean
    clean = strip_post_hoc_labels(clean)
    if mode == "text-only":
        clean.pop("rule_refs_raw", None)
    return clean


def _extract_raw_text_for_dedup(raw_obj: Dict[str, Any]) -> str:
    messages = raw_obj.get("messages")
    parts: List[str] = []
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role", "player") == "player":
                parts.append(str(m.get("text") or ""))
    if not parts and raw_obj.get("text"):
        parts.append(str(raw_obj.get("text") or ""))
    return "\n".join(parts).strip()


def process_batch_file(
    input_path: str,
    output_path: str,
    workers: int = 8,
    rules_path: Optional[str] = None,
    strip_labels: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Обрабатывает JSONL-файл тикетов.

    `strip_labels` сохранён для обратной совместимости (True => mode="strip-labels",
    False => mode="full"); приоритет у явно переданного `mode`.
    """
    if mode is None:
        mode = "strip-labels" if strip_labels else "full"
    if mode not in PREDICT_MODES:
        raise ValueError(f"Unknown predict mode: {mode!r}; expected one of {PREDICT_MODES}")

    t0 = time.perf_counter()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_records: List[Tuple[int, str, Dict[str, Any]]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    obj = {}
            except Exception:
                obj = {}
            ticket_id = str(obj.get("ticket_id") if obj.get("ticket_id") is not None else f"row_{idx + 1}")
            obj = apply_predict_mode(obj, mode)
            raw_records.append((idx, ticket_id, obj))

    # Step 1: Deterministic full-file deduplication pass in chronological/file order
    batch_dedup = Deduplicator()
    dedup_flags: List[bool] = []
    for _, _, obj in raw_records:
        txt = _extract_raw_text_for_dedup(obj)
        dup_hint = obj.get("DUPLICATE_HINT") if obj.get("DUPLICATE_HINT") is not None else obj.get("duplicate_hint")
        is_spam_or_dup, _ = batch_dedup.check_and_record(
            text=txt,
            reporter_hash=obj.get("reporter_hash"),
            created_at=obj.get("created_at"),
            duplicate_hint=bool(dup_hint) if dup_hint is not None else None,
        )
        dedup_flags.append(is_spam_or_dup)

    # Step 2: Concurrent triage processing preserving exact input order
    pipeline = SupportCopilotPipeline(rules_path=rules_path)

    def _worker(args_tuple: Tuple[int, str, Dict[str, Any], bool]) -> Dict[str, Any]:
        idx, ticket_id, obj, dup_flag = args_tuple
        try:
            req = TicketRequest(**obj)
            triage_resp = pipeline.process_ticket(req, dedup_override=dup_flag)
            out_dict = {"ticket_id": ticket_id}
            out_dict.update(triage_resp.to_dict())
            return out_dict
        except Exception:
            fallback = build_fallback_response(
                player_text=_extract_raw_text_for_dedup(obj),
                kind=str(obj.get("kind", "report")) if obj.get("kind") in ("complaint", "report") else "report",
                server_lang=str(obj.get("server_lang") or "ru"),
                reported_nick=obj.get("reported_nick"),
                rule_refs_raw=obj.get("rule_refs_raw"),
                attachments=0,
                images_count=0,
                is_spam_or_dup=dup_flag,
                rag_items=[],
                is_error_fallback=True,
            )
            out_dict = {"ticket_id": ticket_id}
            out_dict.update(fallback.to_dict())
            return out_dict

    tasks = [
        (idx, ticket_id, obj, dedup_flags[i])
        for i, (idx, ticket_id, obj) in enumerate(raw_records)
    ]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(_worker, tasks))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for res in results:
            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    nh_count = sum(1 for r in results if r.get("needs_human") is True)
    auto_count = len(results) - nh_count
    return {
        "total_tickets": len(results),
        "needs_human_count": nh_count,
        "needs_human_rate": round(nh_count / float(max(1, len(results))), 4),
        "auto_closed_count": auto_count,
        "mode": mode,
        "strip_labels": mode != "full",
        "elapsed_sec": round(elapsed, 4),
        "output_path": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch JSONL triage prediction")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--workers", "-w", type=int, default=int(os.getenv("PREDICT_WORKERS", "8")), help="Worker threads")
    parser.add_argument(
        "--mode",
        "-m",
        choices=PREDICT_MODES,
        default=os.getenv("PREDICT_MODE", "strip-labels"),
        help="full | strip-labels (default) | text-only",
    )
    parser.add_argument(
        "--strip-labels",
        action="store_true",
        help="(устаревший алиас) то же, что --mode strip-labels",
    )
    args = parser.parse_args()

    mode = "strip-labels" if args.strip_labels else args.mode
    stats = process_batch_file(
        args.input,
        args.output,
        workers=args.workers,
        mode=mode,
    )
    print(
        f"[predict] Processed {stats['total_tickets']} tickets (mode={stats['mode']}, "
        f"needs_human={stats['needs_human_count']}/{stats['total_tickets']} ({stats['needs_human_rate']:.4f}), "
        f"auto_closed={stats['auto_closed_count']}) -> {stats['output_path']} in {stats['elapsed_sec']}s"
    )


if __name__ == "__main__":
    main()
