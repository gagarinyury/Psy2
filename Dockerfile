FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Системные пакеты при необходимости (компиляция)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Установить зависимости сначала для кэша
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Копируем проект
COPY . /app

# Установка как пакет (setuptools)
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]