#!/usr/bin/env python3
"""Live end-to-end + edge-case test suite cho RAG API (NPU AI080).

Chạy trực tiếp vào hệ thống đang chạy (mặc định http://localhost:18000).
Bao phủ: health, embed/text, embed/file, collections CRUD, query (Phase 1),
transcript embed + query + context + segments (Phase 2) và các edge case
(validation 422/400/404/413, E5 multilingual relevance, sequence_id tăng dần,
window clamp, prefix filter, cleanup).

Usage:
    python scripts/test_live_api.py [--base-url http://localhost:18000]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import uuid

import httpx

# ----------------------------- mini test harness -----------------------------

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[32mPASS\033[0m  {name}")
    else:
        FAIL += 1
        msg = f"{name}" + (f"  -> {detail}" if detail else "")
        FAILURES.append(msg)
        print(f"  \033[31mFAIL\033[0m  {name}" + (f"  -> {detail}" if detail else ""))
    return cond


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def poll(fn, timeout: float = 60.0, interval: float = 1.5):
    """Poll fn() until it returns truthy or timeout. Returns last value."""
    deadline = time.time() + timeout
    val = None
    while time.time() < deadline:
        val = fn()
        if val:
            return val
        time.sleep(interval)
    return val


# ----------------------------- test body -------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:18000")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # unique names để không đụng dữ liệu thật + tự dọn cuối bài
    run = uuid.uuid4().hex[:8]
    doc_coll = f"test-doc-{run}"
    meet_coll = f"meeting-test-{run}"
    bad_meet_coll = f"notmeeting-{run}"

    c = httpx.Client(base_url=base, timeout=60.0)

    # ----- health -----
    section("Health")
    r = c.get("/health")
    check("GET /health -> 200", r.status_code == 200, str(r.status_code))
    check("health body status=ok", r.json().get("status") == "ok", r.text)

    # ----- embed/text validation edge cases -----
    section("embed/text validation")
    r = c.post("/embed/text", json={"source": "s"})  # missing text
    check("missing text -> 422", r.status_code == 422, str(r.status_code))

    r = c.post("/embed/text", json={"text": "", "source": "s"})  # empty text
    check("empty text -> 422", r.status_code == 422, str(r.status_code))

    r = c.post("/embed/text", json={"text": "hello"})  # missing source
    check("missing source -> 422", r.status_code == 422, str(r.status_code))

    # ----- embed/text happy path (VI + EN, multi-chunk) -----
    section("embed/text ingest (multilingual)")
    docs = [
        ("vi-hop", "Cuộc họp ban giám đốc thảo luận về kế hoạch ngân sách quý ba và "
                   "chiến lược mở rộng thị trường miền Nam trong năm nay."),
        ("en-budget", "The board meeting discussed the third quarter budget plan and the "
                      "strategy to expand into the southern market this year."),
        ("vi-kythuat", "Đội kỹ thuật báo cáo tiến độ tích hợp embedding đa ngữ chạy trên "
                       "card tăng tốc NPU Qualcomm AI080 cho hệ thống tìm kiếm ngữ nghĩa."),
    ]
    for src, text in docs:
        r = c.post("/embed/text", json={"text": text, "source": src, "collection": doc_coll})
        check(f"embed/text '{src}' -> 200", r.status_code == 200, r.text[:200])
        if r.status_code == 200:
            b = r.json()
            check(f"  '{src}' response success+doc_id", b.get("success") and bool(b.get("doc_id")), r.text)

    # custom doc_id passthrough
    fixed_id = str(uuid.uuid4())
    r = c.post("/embed/text", json={"text": "Tài liệu có doc_id cố định để kiểm tra xóa theo id.",
                                    "source": "vi-fixed", "collection": doc_coll, "doc_id": fixed_id,
                                    "metadata": {"tag": "unit"}})
    check("embed/text custom doc_id echoed", r.status_code == 200 and r.json().get("doc_id") == fixed_id, r.text[:200])

    # wait until collection has the docs embedded (background NPU task)
    def docs_ready():
        rr = c.get(f"/embed/{doc_coll}/documents")
        if rr.status_code != 200:
            return None
        total = rr.json().get("total", 0)
        return total if total >= 4 else None

    total_docs = poll(docs_ready, timeout=90)
    check("background embed produced >=4 docs", bool(total_docs), f"total={total_docs}")

    # collection now appears in list, vector_size 768
    r = c.get("/embed/collections")
    check("doc collection listed", r.status_code == 200 and doc_coll in r.json().get("collections", []), r.text[:200])

    # ----- query Phase 1 validation -----
    section("query validation")
    r = c.post("/query/", json={"query": "   ", "collection": doc_coll})  # whitespace -> 400 after strip
    check("whitespace query -> 400", r.status_code == 400, str(r.status_code))

    r = c.post("/query/", json={"query": "", "collection": doc_coll})  # empty -> 422 pydantic
    check("empty query -> 422", r.status_code == 422, str(r.status_code))

    r = c.post("/query/", json={"query": "x", "collection": doc_coll, "top_k": 0})
    check("top_k=0 -> 422", r.status_code == 422, str(r.status_code))

    r = c.post("/query/", json={"query": "x", "collection": doc_coll, "top_k": 51})
    check("top_k=51 -> 422", r.status_code == 422, str(r.status_code))

    r = c.post("/query/", json={"query": "x", "collection": doc_coll, "score_threshold": 1.5})
    check("score_threshold>1 -> 422", r.status_code == 422, str(r.status_code))

    r = c.post("/query/", json={"query": "x", "collection": doc_coll, "score_threshold": -0.1})
    check("score_threshold<0 -> 422", r.status_code == 422, str(r.status_code))

    # ----- query Phase 1 relevance (E5 multilingual) -----
    section("query relevance (E5 VI<->EN cross-lingual)")
    r = c.post("/query/", json={"query": "ngân sách quý ba của công ty", "collection": doc_coll, "top_k": 3})
    check("VI budget query -> 200", r.status_code == 200, r.text[:200])
    if r.status_code == 200:
        res = r.json()["results"]
        check("VI budget query returns results", len(res) > 0, str(len(res)))
        if res:
            top = res[0]
            check("top hit is a budget doc (vi-hop/en-budget)",
                  top["source"] in ("vi-hop", "en-budget"), f"top={top['source']} score={top['score']:.3f}")
            check("top score reasonable (>0.7)", top["score"] > 0.7, f"score={top['score']:.4f}")

    # cross-lingual: English query should retrieve the VI budget doc too
    r = c.post("/query/", json={"query": "quarterly budget and market expansion plan",
                                "collection": doc_coll, "top_k": 3})
    if r.status_code == 200 and r.json()["results"]:
        srcs = [x["source"] for x in r.json()["results"]]
        check("EN query retrieves budget docs (cross-lingual)",
              any(s in ("vi-hop", "en-budget") for s in srcs), f"srcs={srcs}")

    # source_filter narrows results
    r = c.post("/query/", json={"query": "embedding NPU", "collection": doc_coll,
                                "top_k": 5, "source_filter": "vi-kythuat"})
    if r.status_code == 200:
        srcs = {x["source"] for x in r.json()["results"]}
        check("source_filter restricts to one source", srcs.issubset({"vi-kythuat"}) and srcs, f"srcs={srcs}")

    # score_threshold filters out everything when set to 0.999
    r = c.post("/query/", json={"query": "nội dung hoàn toàn không liên quan xyz",
                                "collection": doc_coll, "top_k": 5, "score_threshold": 0.999})
    check("high score_threshold -> few/no results", r.status_code == 200 and r.json()["total_found"] == 0,
          r.text[:200])

    # query against non-existent collection: should not 500
    r = c.post("/query/", json={"query": "test", "collection": f"ghost-{run}"})
    check("query ghost collection no 5xx", r.status_code < 500, f"{r.status_code}: {r.text[:150]}")

    # ----- embed/file edge cases -----
    section("embed/file")
    r = c.post("/embed/file", files={"file": ("empty.txt", b"", "text/plain")},
               data={"collection": doc_coll})
    check("empty file -> 400", r.status_code == 400, str(r.status_code))

    r = c.post("/embed/file",
               files={"file": ("bad.txt", b"noi dung hop le", "text/plain")},
               data={"collection": doc_coll, "extra_metadata": "{not json"})
    check("invalid extra_metadata JSON -> 400", r.status_code == 400, str(r.status_code))

    file_text = ("Báo cáo kỹ thuật: hệ thống RAG sử dụng Qdrant làm vector store và mô hình "
                 "embedding đa ngữ multilingual-e5-base 768 chiều chạy trên NPU. " * 30).encode()
    r = c.post("/embed/file",
               files={"file": ("baocao.txt", file_text, "text/plain")},
               data={"collection": doc_coll, "extra_metadata": '{"category":"report"}'})
    check("valid .txt file -> 200", r.status_code == 200, r.text[:200])
    file_doc_id = r.json().get("doc_id") if r.status_code == 200 else None

    # ----- delete document by source / doc_id -----
    section("delete document")
    poll(lambda: c.get(f"/embed/{doc_coll}/documents").json().get("total", 0) >= 5, timeout=60)
    r = c.delete(f"/embed/{doc_coll}/doc/{fixed_id}")
    check("delete by doc_id -> 200", r.status_code == 200, r.text[:200])
    r = c.delete(f"/embed/{doc_coll}/source/vi-kythuat")
    check("delete by source -> 200", r.status_code == 200, r.text[:200])
    # verify gone
    def src_gone():
        rr = c.get(f"/embed/{doc_coll}/documents")
        if rr.status_code != 200:
            return None
        srcs = {d.get("source") for d in rr.json().get("documents", [])}
        return "vi-kythuat" not in srcs
    check("deleted source no longer listed", bool(poll(src_gone, timeout=30)))

    # ----- collections CRUD -----
    section("collections create/delete")
    tmp_coll = f"test-crud-{run}"
    r = c.post("/embed/collections", data={"name": tmp_coll})
    check("create collection -> 2xx", r.status_code < 300, r.text[:200])
    r = c.post("/embed/collections", data={"name": tmp_coll})  # duplicate
    check("create duplicate no 5xx", r.status_code < 500, f"{r.status_code}: {r.text[:150]}")
    r = c.request("DELETE", "/embed/collections", data={"name": tmp_coll})
    check("delete collection -> 2xx", r.status_code < 300, r.text[:200])
    r = c.post("/embed/collections")  # missing name
    check("create without name -> 422", r.status_code == 422, str(r.status_code))

    # ----- transcript prefix validation -----
    section("transcript validation (prefix meeting-)")
    r = c.post(f"/transcript/{bad_meet_coll}/embed",
               json={"speaker": "A", "text": "xin chào"})
    check("embed non-meeting collection -> 400", r.status_code == 400, str(r.status_code))

    r = c.post(f"/transcript/{meet_coll}/embed", json={"speaker": "A"})  # missing text
    check("transcript missing text -> 422", r.status_code == 422, str(r.status_code))

    r = c.post(f"/transcript/{meet_coll}/embed", json={"text": "thiếu speaker"})  # missing speaker
    check("transcript missing speaker -> 422", r.status_code == 422, str(r.status_code))

    # ----- transcript ingest + sequence_id increment -----
    section("transcript ingest (sequence_id tăng dần)")
    lines = [
        ("Alice", "Chúng ta bắt đầu cuộc họp về kế hoạch sản phẩm mới."),
        ("Bob", "Tôi đề xuất ưu tiên tính năng tìm kiếm ngữ nghĩa tiếng Việt."),
        ("Alice", "Đồng ý, đội kỹ thuật sẽ tích hợp embedding e5 trên NPU AI080."),
        ("Bob", "Chi phí phần cứng Qualcomm và tiến độ triển khai thế nào?"),
        ("Carol", "We should also benchmark latency for real-time meeting transcription."),
    ]
    seqs = []
    point_ids = []
    for sp, tx in lines:
        r = c.post(f"/transcript/{meet_coll}/embed", json={"speaker": sp, "text": tx, "lang": "vi"})
        if not check(f"embed transcript '{sp[:1]}' -> 202", r.status_code == 202, r.text[:200]):
            continue
        b = r.json()
        seqs.append(b["sequence_id"])
        point_ids.append(b["point_id"])
        check(f"  context_status pending/disabled", b["context_status"] in ("pending", "disabled"), b.get("context_status"))

    check("sequence_id starts at 1", seqs and seqs[0] == 1, str(seqs))
    check("sequence_id strictly increasing 1..N", seqs == list(range(1, len(seqs) + 1)), str(seqs))
    check("point_ids unique", len(set(point_ids)) == len(point_ids), str(point_ids))

    # wait for transcript vectors to be embedded/searchable
    def seg_ready():
        rr = c.get(f"/transcript/{meet_coll}/segments", params={"from_seq": 1, "limit": 100})
        if rr.status_code != 200:
            return None
        return rr.json().get("count", 0) >= len(lines)
    check("transcript segments embedded", bool(poll(seg_ready, timeout=90)))

    # ----- segments edge cases -----
    section("segments")
    r = c.get(f"/transcript/{meet_coll}/segments", params={"from_seq": 0})
    check("segments from_seq=0 -> 422", r.status_code == 422, str(r.status_code))
    r = c.get(f"/transcript/{meet_coll}/segments", params={"limit": 2000})
    check("segments limit=2000 -> 422", r.status_code == 422, str(r.status_code))
    r = c.get(f"/transcript/{bad_meet_coll}/segments")
    check("segments non-meeting -> 400", r.status_code == 400, str(r.status_code))
    r = c.get(f"/transcript/{meet_coll}/segments", params={"from_seq": 2, "to_seq": 4})
    if check("segments range 2..4 -> 200", r.status_code == 200, r.text[:200]):
        body = r.json()
        got = [s["sequence_id"] for s in body["segments"]]
        check("segments range respects from/to", all(2 <= s <= 4 for s in got) and got, str(got))

    # ----- transcript query -----
    section("query/transcript")
    r = c.post("/query/transcript", json={"query": "test"})  # collection None -> 400
    check("transcript query no collection -> 400", r.status_code == 400, str(r.status_code))
    r = c.post("/query/transcript", json={"query": "test", "collection": bad_meet_coll})
    check("transcript query bad prefix -> 400", r.status_code == 400, str(r.status_code))

    r = c.post("/query/transcript", json={
        "query": "tích hợp embedding trên phần cứng NPU",
        "collection": meet_coll, "top_k": 3, "window_size": 2,
    })
    if check("transcript query -> 200", r.status_code == 200, r.text[:200]):
        body = r.json()
        check("transcript query returns results", body["count"] > 0, str(body["count"]))
        if body["results"]:
            top = body["results"][0]
            check("top transcript hit mentions embedding/NPU",
                  any(k in top["text"].lower() for k in ("embedding", "npu", "e5")),
                  f"top seq={top['sequence_id']} text={top['text'][:60]!r}")
            check("window present (±2)", top.get("window") is not None)
            if top.get("window"):
                wb = len(top["window"]["before"])
                wa = len(top["window"]["after"])
                check("window sizes within clamp (<=5 each)", wb <= 5 and wa <= 5, f"before={wb} after={wa}")

    # window_size clamp: ask for huge window, must be clamped by TRANSCRIPT_MAX_WINDOW_SIZE
    r = c.post("/query/transcript", json={
        "query": "kế hoạch sản phẩm", "collection": meet_coll, "top_k": 1, "window_size": 999,
    })
    if r.status_code == 200 and r.json()["results"]:
        w = r.json()["results"][0].get("window") or {"before": [], "after": []}
        check("oversized window_size clamped (<=5)",
              len(w["before"]) <= 5 and len(w["after"]) <= 5,
              f"before={len(w['before'])} after={len(w['after'])}")

    # speaker_filter
    r = c.post("/query/transcript", json={
        "query": "cuộc họp", "collection": meet_coll, "top_k": 5, "speaker_filter": "Carol",
    })
    if r.status_code == 200 and r.json()["results"]:
        spk = {x["speaker"] for x in r.json()["results"]}
        check("speaker_filter restricts speaker", spk == {"Carol"}, f"speakers={spk}")

    # ----- context endpoint -----
    section("context")
    r = c.get(f"/transcript/{bad_meet_coll}/context")
    check("context non-meeting -> 400", r.status_code == 400, str(r.status_code))
    r = c.get(f"/transcript/meeting-ghost-{run}/context")
    check("context unknown meeting -> 404", r.status_code == 404, str(r.status_code))
    # LLM_PROVIDER=none -> context likely pending/disabled but endpoint should resolve a status or 404
    r = c.get(f"/transcript/{meet_coll}/context")
    check("context latest -> 200 or 404", r.status_code in (200, 404), f"{r.status_code}: {r.text[:150]}")
    if r.status_code == 200:
        check("context_status is a valid enum",
              r.json().get("context_status") in ("pending", "processing", "ready", "failed", "disabled"),
              r.text[:150])

    # ----- cleanup -----
    section("cleanup")
    r = c.request("DELETE", "/embed/collections", data={"name": doc_coll})
    check("cleanup doc collection", r.status_code < 300, r.text[:150])
    r = c.request("DELETE", "/embed/collections", data={"name": meet_coll})
    check("cleanup meeting collection", r.status_code < 300, r.text[:150])

    c.close()

    # ----- summary -----
    print(f"\n{'='*50}")
    print(f"TOTAL: {PASS + FAIL}   \033[32mPASS={PASS}\033[0m   \033[31mFAIL={FAIL}\033[0m")
    if FAILURES:
        print("\nFailures:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
