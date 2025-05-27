FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/LeanModels/Bagel-DFloat11.git /app

# Install Python dependencies
RUN pip3 install --no-cache-dir torch==2.6 torchvision
RUN pip3 install --no-cache-dir packaging
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && \
    pip3 install --no-cache-dir . && \
    cd ..
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install flask

COPY load_weights.py .
COPY server.py .

# Create a script to check and load weights if needed
RUN echo '#!/bin/bash \n\
if [ ! -d "/app/BAGEL-7B-MoT-DF11" ] || [ -z "$(ls -A /app/BAGEL-7B-MoT-DF11)" ]; then \n\
    echo "Model weights not found. Downloading..." \n\
    python3 load_weights.py \n\
fi \n\
python3 server.py \n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port 5000
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT ["/app/start.sh"]
