FROM ghcr.io/astral-sh/uv:0.11.26 AS uv
FROM python:3.11.15-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 cafe
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
# The glibc Linux CPU wheel is separate from the evaluated macOS 0.3.9 build.
# --only-binary prevents an accidental native/GPU source build in this image.
RUN uv export --locked --extra cpu --no-dev --no-emit-project --no-hashes \
    --output-file /tmp/requirements.txt \
    && uv --no-config pip install --system -r /tmp/requirements.txt \
       --no-deps --only-binary llama-cpp-python \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
       --index-strategy unsafe-best-match \
    && uv --no-config pip install --system --no-deps . \
    && mkdir -p /app/models /app/.cache \
    && chown -R cafe:cafe /app
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/cafe-entrypoint
ENV CAFE_HOST=0.0.0.0 PORT=7860 TOKENIZERS_PARALLELISM=false \
    PYTHONHASHSEED=42 GRADIO_ANALYTICS_ENABLED=False HF_HOME=/app/.cache/huggingface
USER cafe
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
ENTRYPOINT ["cafe-entrypoint"]
