# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY .EasyOCR/ /root/.EasyOCR/
COPY . .

ENTRYPOINT ["python", "main.py"]
