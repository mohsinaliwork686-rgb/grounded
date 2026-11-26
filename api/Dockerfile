# The grounded API: ingests the mounted documents, then serves POST /api/ask.
# Both local models are downloaded at build time, so the container runs without internet
# (except for Claude, when ANTHROPIC_API_KEY is set).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/opt/models

WORKDIR /app

# CPU-only PyTorch first: the default wheel brings several GB of CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

COPY requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

# The embedding model and the NLI model, downloaded before grounded's code is copied in, so a
# code change doesn't download them again. These must match DEFAULT_MODEL in embed.py and
# DEFAULT_NLI_MODEL in verify.py; the build checks that below.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-small
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}', device='cpu'); \
CrossEncoder('${NLI_MODEL}', device='cpu')"

COPY pyproject.toml ./
COPY grounded ./grounded
RUN pip install ".[ml]" -c requirements-ml.txt \
    && python -c "\
from grounded.embed import DEFAULT_MODEL; from grounded.verify import DEFAULT_NLI_MODEL; \
assert (DEFAULT_MODEL, DEFAULT_NLI_MODEL) == ('${EMBEDDING_MODEL}', '${NLI_MODEL}'), \
'the Dockerfile downloads different models from the code defaults'"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    GROUNDED_DOCS=/docs

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh \
    && useradd --create-home --uid 10001 grounded
USER grounded

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "grounded.app:app", "--host", "0.0.0.0", "--port", "8000"]
