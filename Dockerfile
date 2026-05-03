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

# HF Spaces requires port 7860
EXPOSE 7860

# Default: Streamlit demo (HF Spaces). docker-compose overrides this for local API use.
CMD ["streamlit", "run", "hf_space/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
