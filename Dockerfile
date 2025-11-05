# Official Python slim image with required build tools for some wheels
FROM python:3.12-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies needed to build some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libffi-dev \
       libssl-dev \
       git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject first so Docker layer caching keeps dependencies cached when code changes
COPY pyproject.toml /app/

# Upgrade pip and install the Python dependencies listed in pyproject.toml's [project].dependencies
# We install the packages directly (avoids relying on a build backend being present in pyproject).
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        "apify-client>=1.12.2" \
        "beautifulsoup4>=4.14.2" \
        "fastapi>=0.121.0" \
        "google-adk>=1.17.0" \
        "google-genai>=1.48.0" \
        "jinja2>=3.1.6" \
        "json-repair>=0.52.4" \
        "langchain>=1.0.3" \
        "langchain-google-genai>=3.0.0" \
        "markdown>=3.10" \
        "networkx>=3.5" \
        "pymongo>=4.14.0" \
        "python-dateutil>=2.9.0.post0" \
        "python-dotenv>=1.1.1" \
        "pytz>=2025.2" \
        "rdflib>=7.4.0" \
        "requests>=2.32.4" \
        "uvicorn>=0.38.0"

# Copy app source
COPY . /app

# Expose port used by uvicorn
EXPOSE ${PORT}

# Use uvicorn to run the FastAPI app. We run the webapp package entrypoint.
# In development you may want to add --reload, but do not use reload in production.
CMD ["python", "main.py"]
