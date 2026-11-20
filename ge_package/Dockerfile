# CPU-only image that runs the GE scripts on synthetic records.
#   docker build -t ge-synthetic .
#   docker run --rm ge-synthetic
#   docker run --rm ge-synthetic python bench_latency.py
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements_pinned.txt .
# the CPU build of torch comes from the PyTorch index, its dependencies from PyPI
RUN pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
    && pip install -r requirements_pinned.txt

COPY ge_lib.py ge_build.py ge_train.py make_synthetic_ge.py paired_stats.py \
     bench_latency.py run_synthetic.sh ./

CMD ["sh", "run_synthetic.sh"]
