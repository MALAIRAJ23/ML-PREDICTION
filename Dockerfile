FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Initialize database and train models during build
RUN python setup_fixed.py

EXPOSE 5000

CMD ["python", "app.py"]
