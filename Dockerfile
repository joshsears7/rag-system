FROM python:3.11-slim

# System deps for PDF/DOCX processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer — split so model download layer is separate)
COPY hf_space/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models into the image so cold starts don't re-download them.
# These layers are cached by Docker and reused on every container restart.
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application code (after model download so code changes don't invalidate model cache)
COPY . .

# Create data directories
RUN mkdir -p /app/data/chroma_db /app/data

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["streamlit", "run", "hf_space/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
