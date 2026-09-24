FROM docker.novin.cloud/python:3.12-slim-bookworm

WORKDIR /app

# Use Iranian Debian mirrors only.
# The base image is Debian Bookworm.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf '%s\n' \
    'deb https://repo.mirror.famaserver.com/debian bookworm main' \
    'deb https://repo.mirror.famaserver.com/debian bookworm-updates main' \
    'deb https://mirror.mobinhost.com/debian-security bookworm-security main' \
    > /etc/apt/sources.list

# Install Firefox ESR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        firefox-esr \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Force pip to use the Iranian PyPI mirror.
ENV PIP_INDEX_URL=https://mirror.novin.cloud/artifactory/api/pypi/pypi/simple/ \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER=1

COPY requirements.txt .

RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]