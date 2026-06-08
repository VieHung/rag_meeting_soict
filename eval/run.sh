#!/bin/bash
# Chạy RAGAS focused (subset=4, metrics: answer_relevancy + faithfulness).
# Đây là subset an toàn cho LM Studio model 8K context (gemma-4-e4b).
# Để chạy retrieval + RAGAS kết hợp, dùng run_full.sh.
set -e
unset OPENROUTER_API_KEY
cd /mnt/c/Users/navis/hungtv/rag_base
PY=/mnt/c/Users/navis/hungtv/rag_base/venv/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
$PY -m eval.run_ragas --ragas-subset 4 --ragas-metrics answer_relevancy,faithfulness --out /mnt/c/Users/navis/hungtv/rag_base/eval/results/metrics_ragas_focused.json 2>&1 | tee -a /mnt/c/Users/navis/hungtv/rag_base/eval/eval_run.log

