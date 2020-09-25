FROM python:3.7.3-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER mearlboro@protonmail.com

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

## Do not edit if you have a requirements.txt
# Running copy before pip run caches python packages when docker building
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet
