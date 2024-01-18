FROM python:3.9-slim

RUN mkdir /front


WORKDIR /front


RUN pip install requests numpy pandas matplotlib scikit-learn streamlit


COPY . /front

EXPOSE 8501

CMD ["streamlit", "run", "f_main.py"]


