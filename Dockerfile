FROM python:3.11-slim

# System deps for PDF/DOCX processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for ChromaDB persistence
RUN mkdir -p /app/data/chroma_db

# Expose FastAPI port
EXPOSE 8000

# Default: run API server
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
