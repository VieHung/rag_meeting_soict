#!/usr/bin/env bash
#
# build_qpc.sh — Export embedding model HuggingFace → ONNX → compile QPC cho Qualcomm AI080.
#
# Chạy MỘT LẦN trên host AI080 (đã cài Qualcomm Apps/Platform SDK: qaic-exec, qaic).
# Sản phẩm: thư mục QPC → trỏ EMBEDDING_QPC_PATH tới đó khi EMBEDDING_BACKEND=qaic.
#
#   pip install -r requirements-build.txt   # optimum, onnx, torch (chỉ cho bước export)
#   bash scripts/build_qpc.sh
#
# LƯU Ý: cú pháp/flag của qaic-exec phụ thuộc phiên bản SDK trên AI080 — chỉnh lại
# SEQ_LEN, NUM_CORES, precision theo cấu hình thực tế. SEQ_LEN PHẢI khớp
# EMBEDDING_MAX_SEQ_LEN trong .env.
set -euo pipefail

MODEL="${EMBEDDING_MODEL:-intfloat/multilingual-e5-base}"
SEQ_LEN="${EMBEDDING_MAX_SEQ_LEN:-128}"
NUM_CORES="${AIC_NUM_CORES:-4}"
BATCH="${EMBED_BATCH:-1}"

WORK_DIR="${WORK_DIR:-./qpc_build}"
ONNX_DIR="${WORK_DIR}/onnx"
QPC_OUT="${QPC_OUT:-./qpc/e5-base}"

# Chú ý: KHÔNG tạo sẵn QPC_OUT — qaic-exec yêu cầu -aic-binary-dir CHƯA tồn tại.
mkdir -p "${ONNX_DIR}" "$(dirname "${QPC_OUT}")"

# HuggingFace cache: nếu cache mặc định (~/.cache/huggingface) không ghi được
# (vd bị tạo bởi root qua sudo → mode 700), chuyển sang thư mục ghi được trong WORK_DIR
# để tránh PermissionError khi đọc token / tải model.
DEFAULT_HF="${HOME}/.cache/huggingface"
if [ -z "${HF_HOME:-}" ] && { [ ! -w "${DEFAULT_HF}" ] && [ -e "${DEFAULT_HF}" ]; }; then
  export HF_HOME="${WORK_DIR}/hf_cache"
  echo "    [i] ~/.cache/huggingface không ghi được → dùng HF_HOME=${HF_HOME}"
  mkdir -p "${HF_HOME}"
fi

echo "==> [1/2] Export ${MODEL} sang ONNX (seq_len=${SEQ_LEN})"
# Encoder-only feature-extraction → output last_hidden_state. Shape tĩnh để compile QAIC.
optimum-cli export onnx \
  --model "${MODEL}" \
  --task feature-extraction \
  --sequence_length "${SEQ_LEN}" \
  --batch_size "${BATCH}" \
  "${ONNX_DIR}"

ONNX_FILE="${ONNX_DIR}/model.onnx"
echo "    ONNX: ${ONNX_FILE}"

echo "==> [2/2] Compile ONNX → QPC bằng qaic-exec (cores=${NUM_CORES}, fp16)"
# Tự dò qaic-exec: ưu tiên PATH, nếu không có thì lấy ở vị trí chuẩn của SDK.
QAIC_EXEC="${QAIC_EXEC:-$(command -v qaic-exec || true)}"
if [ -z "${QAIC_EXEC}" ] && [ -x /opt/qti-aic/exec/qaic-exec ]; then
  QAIC_EXEC=/opt/qti-aic/exec/qaic-exec
fi
if [ -z "${QAIC_EXEC}" ]; then
  echo "ERROR: không tìm thấy qaic-exec. Cài Qualcomm Apps SDK hoặc đặt QAIC_EXEC=/duong/dan/qaic-exec" >&2
  exit 127
fi
echo "    qaic-exec: ${QAIC_EXEC}"

# qaic-exec yêu cầu thư mục output chưa tồn tại → dọn sạch nếu có (chỉ là QPC build artifact).
rm -rf "${QPC_OUT}"

# Flag điển hình của Qualcomm Cloud AI SDK; điều chỉnh theo SDK trên AI080.
"${QAIC_EXEC}" \
  -m="${ONNX_FILE}" \
  -aic-hw -aic-hw-version=2.0 \
  -aic-num-cores="${NUM_CORES}" \
  -convert-to-fp16 \
  -onnx-define-symbol=batch_size,"${BATCH}" \
  -onnx-define-symbol=sequence_length,"${SEQ_LEN}" \
  -compile-only \
  -aic-binary-dir="${QPC_OUT}"

echo "==> Done. QPC: ${QPC_OUT}"
echo "    Đặt trong .env:  EMBEDDING_BACKEND=qaic"
echo "                     EMBEDDING_QPC_PATH=${QPC_OUT}"
echo "                     EMBEDDING_MAX_SEQ_LEN=${SEQ_LEN}"
