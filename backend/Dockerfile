FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY . .

# Download models from Hugging Face
RUN python download_models.py

# Expose port for Hugging Face
EXPOSE 7860

# Start Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]