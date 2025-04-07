FROM python:3.10-slim

WORKDIR /app

# Install system dependencies with clean up
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-distutils \
    wget \
    chromium \
    chromium-driver \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
