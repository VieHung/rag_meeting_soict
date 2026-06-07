# Image cho rag_api chạy embedding trên NPU Qualcomm AI080 (backend qaic).
# Base Ubuntu 22.04 để KHỚP glibc/libstdc++ với native lib SDK (qaicrt cpython-310).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# python3.10 (mặc định Ubuntu 22.04) + lib runtime cho native .so của SDK Qualcomm.
# libpci3/libudev1/libzstd1/libcrypt1: deps của qaicrt/qaiccc/libQAic.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libgomp1 libstdc++6 ca-certificates \
        libpci3 libudev1 libzstd1 libcrypt1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip3 install --no-cache-dir -r requirements-runtime.txt

COPY . .

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 1904
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1904", "--workers", "1"]
