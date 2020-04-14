FROM python:3.7-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir code/

WORKDIR code/

COPY testsite/ code/

COPY requirements.txt code/

# needed to be open to the web
EXPOSE 8000

# needed to communicate with rabbit mq server
EXPOSE 5672

RUN pip install -r code/requirements.txt

CMD python ./code/manage.py runserver 0.0.0.0:8000

# google cloud run url: gcr.io/testapp-273803/testapp
