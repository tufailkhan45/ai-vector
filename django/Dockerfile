 # pull official base image
 FROM python:3.10-slim-buster

 # set work directory
 WORKDIR /usr/src/
 
 # set environment variables
 ENV PYTHONDONTWRITEBYTECODE 1
 ENV PYTHONUNBUFFERED 1
 
 # install mysql dependencies
 RUN apt-get update
 RUN apt-get install gcc default-libmysqlclient-dev -y
 
 # install dependencies
 RUN pip install -U pip setuptools wheel
 RUN pip install --upgrade pip
 COPY requirements.txt .
 RUN pip install -r requirements.txt --no-cache-dir
 
 # copy project
 COPY . .