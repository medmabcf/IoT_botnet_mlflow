FROM python:3.9-slim

RUN mkdir /app

WORKDIR /app

RUN pip install mlflow numpy pandas matplotlib requests scikit-learn fastapi[all]

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
