FROM ubuntu:bionic

MAINTAINER Jack Hay jack@reiform.com

WORKDIR /
RUN apt-get update -y --fix-missing
RUN apt-get install -y python3.7-dev python3.7 python3-pip pkg-config wget curl
RUN wget https://golang.org/dl/go1.17.6.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && tar -C /usr/local -xzf go1.17.6.linux-amd64.tar.gz
RUN python3.7 -m pip install Cython

RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt install -y nodejs

WORKDIR /
RUN mkdir mynah
WORKDIR /mynah

COPY api/ api/
COPY python/ python/
COPY frontend/ frontend/
COPY Makefile Makefile
COPY run.sh run.sh

ENV GOOS=linux
ENV PKG_CONFIG_PATH=/mynah/python
ENV GOPATH=$HOME/go
ENV XDG_CACHE_HOME=/tmp/.cache
RUN make GO=/usr/local/go/bin/go all

EXPOSE 8080

CMD ["./run.sh"]
