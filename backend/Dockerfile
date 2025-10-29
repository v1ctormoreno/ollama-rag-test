FROM python:3.11-slim


WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Para pydantic v2 locales correctos y compresión
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*


COPY app.py /app/app.py


EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
