# Trạng thái hệ thống — 2026-06-07 19:37

## Hoàn thành
- ✅ `rag_base/INVENTORY.md` (12 sections + §13 evaluation) — tài liệu đầy đủ
- ✅ `rag_base/eval/REPORT.md` — báo cáo đánh giá chi tiết
- ✅ `rag_base/eval/metrics.json` — retrieval + RAGAS, retrieval ĐÃ SỬA (hybrid thật sự chạy)
- ✅ `rag_base/eval/raw_answers.json` — answers + contexts cho backfill
- ✅ `rag_base/eval/metrics_ragas_focused.json` — kết quả RAGAS focused (4 rows × 2 metrics)
- ✅ 3 bug đã sửa trong `rag_base`:
  1. `rag_base/eval/retrieval_eval.py` — bỏ `__del__`, dùng `with`-statement
  2. `rag_base/eval/run_ragas.py:74-78` — `MEETING_ID="meeting-bt2-eval"`
  3. `rag_base/rag_server/app/config.py:61` — `extra = "ignore"` cho pydantic-settings

## Cấu trúc eval/
```
eval/
├── __init__.py
├── gold_dataset.py           # 12 docs + 10 transcript Q&A
├── pipeline.py                # RAGPipeline (in-memory Qdrant + production modules)
├── retrieval_eval.py          # Hit@K, MRR, TokenR/P, latency (no LLM)
├── run_ragas.py               # RAGAS LLM judge (focused subset)
├── run_eval.py                # main runner (--phase, --ragas-subset, --ragas-metrics)
├── run.sh / run_retrieval.sh / run_full.sh  # WSL wrappers
├── REPORT.md                  # ★ báo cáo chi tiết
├── eval_run.log               # append-only log
├── results/
│   ├── metrics.json           # ★ retrieval (docs + transcript, 2 configs) + RAGAS
│   ├── raw_answers.json
│   └── metrics_ragas_focused.json
```

## Kết quả cuối
| Metric | Pure vector | Hybrid BM25+vector |
|---|---|---|
| Docs Hit@5 | 0.5833 | **0.6667** (+0.083) |
| Docs MRR | 0.3819 | **0.3986** |
| Docs TokenRecall | 0.7410 | **0.7925** |
| Transcript Hit@3 | 1.0000 | 1.0000 |
| Transcript MRR | 0.8833 | **0.9333** |
| Transcript TokenRecall | 0.9653 | **1.0000** |
| Latency p50 | ~15 ms | ~14 ms |

RAGAS (subset 4, judge=gemma-4-e4b):
- docs: answer_relevancy=0.82, faithfulness=None (8K context quá nhỏ)
- transcript: answer_relevancy=0.83, faithfulness=None

## Cấu hình môi trường cần cho lần chạy tới
- **LM Studio**: `http://192.168.240.1:1234/v1`, model `google/gemma-4-e4b` (8K) — duy nhất đang load được
- **Qdrant server**: KHÔNG cần (dùng `:memory:`)
- **Redis**: KHÔNG cần
- **WSL Ubuntu**: venv tại `C:\Users\navis\hungtv\rag_base\venv` (Python 3.12, WSL symlinks)
- **Env cần unset trước khi chạy**: `unset OPENROUTER_API_KEY` (script đã làm)

## Cách chạy lại
```bash
# Retrieval (nhanh, ~30s)
wsl -d Ubuntu bash /mnt/c/Users/navis/hungtv/rag_base/eval/run_retrieval.sh

# Full eval (retrieval + RAGAS subset 4, ~12 phút)
wsl -d Ubuntu bash /mnt/c/Users/navis/hungtv/rag_base/eval/run_full.sh

# Chỉ RAGAS focused (answer_relevancy + faithfulness, subset 4, ~10 phút)
wsl -d Ubuntu bash /mnt/c/Users/navis/hungtv/run_ragas_focused.sh
```

## TODO (chưa làm / có thể làm tiếp)
- [ ] Bật rerank (`RERANK_PROVIDER=local`, `BAAI/bge-reranker-v2-m3`) — cần GPU/đủ RAM
- [ ] Load model ≥32K context (gemma-4-26b hoặc gpt-oss-20b) để chạy full 4 RAGAS metrics
- [ ] Mở rộng gold dataset ≥30 câu/phase (đang 12+10)
- [ ] Cải thiện `_is_relevant` trong retrieval_eval: dùng cosine ≥ 0.7 thay vì token Jaccard
- [ ] Audit các singleton Pydantic khác trong rag_server (vd `get_reranker()`) xem có bug `__del__` tương tự
- [ ] Cho phép cấu hình `meeting_id` filter qua env trong `TranscriptStore`
- [ ] Dọn file tạm: `test_*.sh`, `check_*.sh`, `list_models.sh`, `install_ragas*.sh`, `fix_lc.sh`, `test_lmstudio*.sh` trong `C:\Users\navis\hungtv\`
- [ ] Đổi `LLM_API_KEY` plaintext trong `STATE.md` cũ (security)

## Files KHÔNG còn cần (có thể xoá)
- `C:\Users\navis\hungtv\test_fuse.sh`
- `C:\Users\navis\hungtv\test_hybrid_log.sh`
- `C:\Users\navis\hungtv\test_hybrid_real.sh`
- `C:\Users\navis\hungtv\test_del.sh`
- `C:\Users\navis\hungtv\test_setattr.sh`
- `C:\Users\navis\hungtv\check_pydantic.sh`
- `C:\Users\navis\hungtv\check_hybrid.sh`
- `C:\Users\navis\hungtv\test_wsl.sh` (cũ)
- `C:\Users\navis\hungtv\check_deps.sh` (cũ)
- `C:\Users\navis\hungtv\check_qdrant_memory.sh` (cũ)
- `C:\Users\navis\hungtv\test_lmstudio.sh` (cũ)
- `C:\Users\navis\hungtv\list_models.sh` (cũ)
- `C:\Users\navis\hungtv\test_models.sh` (cũ)
- `C:\Users\navis\hungtv\fix_lc.sh` (cũ)
- `C:\Users\navis\hungtv\install_ragas*.sh` (cũ)
- `C:\Users\navis\hungtv\test_lmstudio_detailed.sh` (cũ)
