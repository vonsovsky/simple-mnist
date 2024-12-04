FROM python:3.9.6-slim-buster

ARG PROJECT_NAME=simple-mnist

WORKDIR /root/$PROJECT_NAME

ADD requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD python main.py
