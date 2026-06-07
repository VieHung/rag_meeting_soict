#!/usr/bin/env bash
# Entrypoint rag_api (backend NPU qaic).
# qaicrt/qaiccc/QAicApi_pb2 nạp qua PYTHONPATH từ SDK mount (set ở docker-compose).
# Gói python `qaic` (pure-python) cài từ wheel trong SDK đã mount nếu chưa có — idempotent.
set -e

if ! python3 -c "import qaic" 2>/dev/null; then
  WHL=$(ls /opt/qti-aic/dev/lib/x86_64/qaic-*-py3-none-any.whl 2>/dev/null | head -1)
  if [ -n "$WHL" ]; then
    echo "[entrypoint] Cài qaic runtime: $WHL"
    pip3 install --no-deps --no-index "$WHL"
  else
    echo "[entrypoint] CẢNH BÁO: không thấy wheel qaic trong /opt/qti-aic (đã mount SDK chưa?)" >&2
  fi
fi

# Kiểm tra nhanh native lib (không chặn khởi động, chỉ log để dễ chẩn đoán).
python3 -c "import qaicrt, qaiccc" 2>/dev/null \
  && echo "[entrypoint] qaicrt/qaiccc OK" \
  || echo "[entrypoint] CẢNH BÁO: chưa import được qaicrt/qaiccc — kiểm tra PYTHONPATH/LD_LIBRARY_PATH + mount SDK" >&2

exec "$@"
