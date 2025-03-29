# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model files (could be done at runtime instead)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')"

# Create necessary directories
RUN mkdir -p data/guidelines data/literature results

# Set up entrypoint
ENTRYPOINT ["python", "mcrag.py"]

# Default command
CMD ["--help"]