#!/bin/bash
set -e
unset OPENROUTER_API_KEY
cd /mnt/c/Users/navis/hungtv/rag_base
PY=/mnt/c/Users/navis/hungtv/rag_base/venv/bin/python
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
$PY -m eval.run_eval --phase retrieval 2>&1 | tee -a /mnt/c/Users/navis/hungtv/rag_base/eval/eval_run.log
