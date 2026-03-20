FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY homer_core/ ./homer_core/
COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
