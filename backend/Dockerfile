FROM python:3.9-slim

RUN mkdir /backend
RUN mkdir /ml


RUN apt update && apt upgrade -y && apt install -y git

RUN pip install mlflow numpy pandas matplotlib scikit-learn fastapi[all]

COPY ./backend /backend
COPY ./ml /ml

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
