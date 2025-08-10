# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Command to run the app with Gunicorn & Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--bind", "0.0.0.0:8000"]