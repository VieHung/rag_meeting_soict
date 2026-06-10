"""Sinh REPORT_COMPREHENSIVE.md từ eval/results/comprehensive.json.

Đọc kết quả benchmark toàn diện (retrieval matrix + RAGAS 4 metrics × config)
và render thành báo cáo Markdown dễ đọc. Xử lý an toàn metric None/NaN.

Chạy:
    python -m eval.gen_report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "comprehensive.json"
OUT = ROOT / "REPORT_COMPREHENSIVE.md"

METRIC_ORDER = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]


def fmt(v: Optional[float]) -> str:
    return "NaN" if v is None else f"{v:.4f}"


def delta(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main() -> None:
    data: Dict[str, Any] = json.loads(SRC.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    retr = data.get("retrieval", {})
    ragas = data.get("ragas", {})

    L = []
    L.append("# Báo cáo benchmark RAG TOÀN DIỆN — `rag_base/rag_server`\n")
    L.append("> Sinh tự động từ `eval/results/comprehensive.json` bởi `eval/gen_report.py`.\n")
    L.append("## 0. Cấu hình\n")
    L.append(f"- **Model (answer + RAGAS judge):** `{cfg.get('model')}`")
    L.append(f"- **Embedding:** `{cfg.get('embedding_model')}` (dim 384)")
    L.append(f"- **Hybrid weights:** vector={cfg.get('hybrid_vector_weight')} / term={cfg.get('hybrid_term_weight')}")
    L.append(f"- **top_k:** docs={cfg.get('top_k_docs')}, transcript={cfg.get('top_k_transcript')}")
    L.append(f"- **RAGAS context cap:** {cfg.get('max_contexts')} contexts × {cfg.get('max_context_chars')} chars")
    L.append(f"- **Metrics:** {', '.join(cfg.get('metrics', []))}")
    L.append("- **Qdrant:** in-memory (`:memory:`) — production code path (embedding/search/fuse).\n")

    # --- Retrieval ---
    L.append("## 1. Retrieval (deterministic, no LLM)\n")
    L.append("| Phase | Config | n | Hit@K | MRR | TokenRecall | TokenPrec | p50 (ms) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for phase, rows in retr.items():
        for r in rows:
            a = r["aggregate"]
            L.append(f"| {phase} | {r['config']} | {a['n']} | {a['hit_rate_at_k']:.4f} | "
                     f"{a['mrr']:.4f} | {a['mean_recall_at_k']:.4f} | {a['mean_precision_at_k']:.4f} | "
                     f"{a['latency_p50_ms']:.1f} |")
    L.append("")
    # Per-kind cho hybrid.
    for phase, rows in retr.items():
        hyb = next((r for r in rows if r["config"] == "hybrid"), None)
        if not hyb:
            continue
        pk = hyb["aggregate"].get("per_kind", {})
        if not pk:
            continue
        L.append(f"### 1.{'1' if phase=='docs' else '2'} Per-kind — {phase} (hybrid)\n")
        L.append("| Kind | n | Hit@K | MRR | TokenRecall | TokenPrec |")
        L.append("|---|---|---|---|---|---|")
        for kind, d in pk.items():
            L.append(f"| {kind} | {d['n']} | {d['hit_rate_at_k']:.4f} | {d['mrr']:.4f} | "
                     f"{d['token_recall']:.4f} | {d['token_precision']:.4f} |")
        L.append("")

    # --- RAGAS ---
    L.append("## 2. RAGAS (LLM judge — full dataset)\n")
    if not ragas:
        L.append("_(Chưa có kết quả RAGAS — chạy `run_comprehensive.py` không kèm `--skip-ragas`.)_\n")
    else:
        for phase in ("docs", "transcript"):
            L.append(f"### 2.{'1' if phase=='docs' else '2'} {phase}\n")
            L.append("| Metric | pure_vector | hybrid | Δ (hybrid−pure) |")
            L.append("|---|---|---|---|")
            pv = ragas.get("pure_vector", {}).get(phase, {}).get("metrics", {})
            hy = ragas.get("hybrid", {}).get(phase, {}).get("metrics", {})
            for m in METRIC_ORDER:
                if m in pv or m in hy:
                    L.append(f"| {m} | {fmt(pv.get(m))} | {fmt(hy.get(m))} | {delta(pv.get(m), hy.get(m))} |")
            # n + duration
            npv = ragas.get("pure_vector", {}).get(phase, {})
            L.append(f"\n_n={npv.get('n','?')}; thời gian pure={npv.get('duration_sec','?')}s, "
                     f"hybrid={ragas.get('hybrid',{}).get(phase,{}).get('duration_sec','?')}s_\n")

    # --- Notes ---
    L.append("## 3. Ghi chú\n")
    L.append("- Metric **NaN** = RAGAS judge không tách được statement / vượt context / lỗi parse "
             "(xem `eval/comprehensive_run.log`). `mean` bỏ qua NaN.")
    L.append("- Retrieval dùng `_is_relevant` token-Jaccard ≥ 0.3 → có thể **under-estimate** với "
             "câu paraphrase nặng (xem REPORT.md §3.3).")
    L.append("- Raw answers lưu ở `eval/results/raw_answers_{config}.json` để backfill metric "
             "mà không cần gọi lại LLM.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
