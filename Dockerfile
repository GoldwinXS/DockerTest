FROM python:3.7

RUN mkdir /testapp

WORKDIR /mkdir

ADD testsite/testsite .

RUN pip install -r requirements.txt
