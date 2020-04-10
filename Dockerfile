FROM python:3.7-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN mkdir code/

WORKDIR code/

COPY testsite/ code/

COPY requirements.txt code/

EXPOSE 8000

RUN pip install -r code/requirements.txt

CMD [ "python", "./code/manage.py", "runserver", "0.0.0.0:8000"]
