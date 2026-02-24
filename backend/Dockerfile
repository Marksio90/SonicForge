FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for Python packages and PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core build tools
    gcc \
    g++ \
    make \
    pkg-config \
    python3-dev \
    # FFmpeg runtime and development libraries (required for PyAV)
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    # Audio processing libraries
    libsndfile1 \
    gcc \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install torch from PyTorch CPU index first (not available on PyPI)
# torch 2.1.x was removed from the CPU index; minimum available is 2.2.0+cpu
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Then install the package dependencies.
# Use a constraints file so audiocraft's stale torch==2.1.0 pin does not
# downgrade the torch we just installed.
COPY pyproject.toml .
RUN printf 'torch==2.6.0\ntorchaudio==2.6.0\n' > /tmp/torch_constraints.txt && \
    pip install --no-cache-dir -e ".[dev]" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -c /tmp/torch_constraints.txt

# Copy application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
