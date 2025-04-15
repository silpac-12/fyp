# Use an official Python base image
FROM python:3.10.13-slim

# Set working directory in the container
WORKDIR /app

# 🛠️ Install system dependencies required by LightGBM & others
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Launch.py", "--server.port=8501", "--server.enableCORS=false"]
