FROM python:3.7.3-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN mkdir /pylensing-workspace
COPY ./ /pylensing-workspace
WORKDIR /pylensing-workspace

RUN pip install -r requirements.txt
RUN echo Please execute setup_docker.sh

