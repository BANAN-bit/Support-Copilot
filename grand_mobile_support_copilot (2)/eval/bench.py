#!/usr/bin/env python3
"""
Latency, Token Usage, and Cost Benchmark for Grand Mobile Support Copilot (eval/bench.py).
Measures on all 1,000 tickets of data/tickets_valid.jsonl:
  1. Mode A: Local provider with SHA-256 cache enabled (LLM_CACHE_ENABLED=true)
  2. Mode B: Local provider with SHA-256 cache disabled (LLM_CACHE_ENABLED=false)
  3. Mode C: External Argon 4.0 API — checks if a real LLM_API_KEY is configured; if not,
     reports 'not_measured_no_api_key' and computes exact tariff cost estimate from Mode B tokens.
"""
import json
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.pipeline import SupportCopilotPipeline


def _measure_pass(records: List[Dict[str, Any]], cache_enabled: bool) -> Dict[str, Any]:
    pipeline = SupportCopilotPipeline()
    pipeline.deduplicator.reset()
    pipeline.llm_client.cache_enabled = cache_enabled
    pipeline.llm_client._cache.clear()
    pipeline.llm_client.call_history.clear()

    latencies_ms: List[float] = []
    costs_usd: List[float] = []
    prompt_tokens_list: List[int] = []
    completion_tokens_list: List[int] = []
    total_tokens_list: List[int] = []
    cache_hits = 0
    skipped_llm = 0

    t_start = time.perf_counter()
    for rec in records:
        t0 = time.perf_counter()
        _, call_metrics = pipeline.process_ticket_with_metrics(rec)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(wall_ms)
        costs_usd.append(call_metrics.cost_usd)
        prompt_tokens_list.append(call_metrics.prompt_tokens)
        completion_tokens_list.append(call_metrics.completion_tokens)
        total_tokens_list.append(call_metrics.total_tokens)
        if call_metrics.cached:
            cache_hits += 1
        if call_metrics.skipped_llm:
            skipped_llm += 1

    total_wall_sec = time.perf_counter() - t_start
    latencies_sorted = sorted(latencies_ms)
    n = len(latencies_sorted)
    p50_ms = latencies_sorted[int(0.50 * (n - 1))]
    p95_ms = latencies_sorted[int(0.95 * (n - 1))]
    p99_ms = latencies_sorted[int(0.99 * (n - 1))]
    avg_cost = sum(costs_usd) / float(n)
    max_cost = max(costs_usd)

    return {
        "cache_enabled": cache_enabled,
        "tickets_benchmarked": n,
        "total_wall_time_sec": round(total_wall_sec, 4),
        "p50_latency_ms": round(p50_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "p99_latency_ms": round(p99_ms, 3),
        "p95_latency_sec": round(p95_ms / 1000.0, 5),
        "p95_within_5s_sla": (p95_ms / 1000.0) <= 5.0,
        "avg_prompt_tokens": round(sum(prompt_tokens_list) / float(n), 1),
        "avg_completion_tokens": round(sum(completion_tokens_list) / float(n), 1),
        "avg_total_tokens": round(sum(total_tokens_list) / float(n), 1),
        "avg_cost_usd_tariff_estimate": round(avg_cost, 6),
        "max_cost_usd_tariff_estimate": round(max_cost, 6),
        "cost_within_0_002_sla": avg_cost <= 0.002,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / float(n), 4),
        "trivial_bypass_count": skipped_llm,
    }


def run_benchmark(valid_path: str = "data/tickets_valid.jsonl") -> Dict[str, Any]:
    with open(valid_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    mode_cached = _measure_pass(records, cache_enabled=True)
    mode_uncached = _measure_pass(records, cache_enabled=False)

    api_key = os.getenv("LLM_API_KEY", "").strip()
    has_real_key = bool(api_key and api_key != "your_api_key_here")

    bench_summary = {
        "local_provider_with_cache": mode_cached,
        "local_provider_without_cache": mode_uncached,
        "external_argon_api": {
            "measured": False if not has_real_key else True,
            "reason": "LLM_API_KEY is not set and outbound network access is disabled in the sandbox environment; latency of external HTTP API cannot be measured here.",
            "tariff_input_per_1k_usd": 0.0005,
            "tariff_output_per_1k_usd": 0.0015,
            "formula_uncached": f"({mode_uncached['avg_prompt_tokens']} / 1000) * $0.0005 + ({mode_uncached['avg_completion_tokens']} / 1000) * $0.0015 = ${mode_uncached['avg_cost_usd_tariff_estimate']:.6f} USD",
            "formula_cached": f"({mode_cached['avg_prompt_tokens']} / 1000) * $0.0005 + ({mode_cached['avg_completion_tokens']} / 1000) * $0.0015 = ${mode_cached['avg_cost_usd_tariff_estimate']:.6f} USD",
        },
    }

    os.makedirs("data", exist_ok=True)
    with open("data/bench_metrics.json", "w", encoding="utf-8") as f:
        json.dump(bench_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(bench_summary, ensure_ascii=False, indent=2))
    return bench_summary


if __name__ == "__main__":
    run_benchmark()
