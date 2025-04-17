FROM python:3.9-slim

WORKDIR /app

# Cài đặt các dependencies cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements trước để tận dụng Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ code
COPY . .

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=sentiment_analysis
ENV DB_USER=postgres
ENV DB_PASSWORD=password

# Mở cổng 8501 cho Streamlit
EXPOSE 8501

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app.py"]