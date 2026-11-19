FROM eclipse-temurin:21-jre-jammy AS java-runtime

FROM python:3.10-slim-bookworm

ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH \
    HF_HOME=/tmp/huggingface \
    PYTHONUNBUFFERED=1

COPY --from=java-runtime /opt/java/openjdk /opt/java/openjdk

# Torch is only pulled in because pyserini requires it; BM25 search runs on the
# JVM. Install the CPU wheel first so pip does not drag in the multi-gigabyte
# CUDA runtime. 2.14.1 is the newest CPU wheel published for CPython 3.10.
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu" \
    && python -m pip install --no-cache-dir "pyserini==1.2.0" "huggingface_hub==0.36.0" \
    && mkdir -p /app /data/sessions /data/indexes \
    && chmod -R g=u /app /data

WORKDIR /app
COPY scripts/domain_backend.py /app/domain_backend.py

EXPOSE 8093
CMD ["python", "/app/domain_backend.py", "--workload", "research_deep_search", "--port", "8093", "--research-mode", "browsecompplus", "--browsecomp-index-path", "/data/indexes/bm25"]
