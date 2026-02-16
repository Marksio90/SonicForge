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
    libsndfile1-dev \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
