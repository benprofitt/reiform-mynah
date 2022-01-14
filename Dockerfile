FROM ubuntu:bionic

MAINTAINER Jack Hay "jack@reiform.com

WORKDIR /
RUN apt-get update -y --fix-missing
RUN apt-get install -y python3.7-dev python3.7 python3-pip pkg-config wget
RUN wget https://golang.org/dl/go1.16.5.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && tar -C /usr/local -xzf go1.16.5.linux-amd64.tar.gz
RUN python3.7 -m pip install Cython && \
    python3.7 -m pip install numpy && \
    python3.7 -m pip install pandas && \
    python3.7 -m pip install scikit-learn

WORKDIR /
RUN mkdir mynah
WORKDIR /mynah

COPY Makefile Makefile
COPY api/ api/
COPY run.sh run.sh

ENV PKG_CONFIG_PATH=/mynah/python
RUN make GO=/usr/local/go/bin/go mynah

EXPOSE 8080

CMD ["./run.sh"]
