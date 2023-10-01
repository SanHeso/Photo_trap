FROM debian:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget vim python3-pip python3-venv -y
RUN apt-get install build-essential zlib1g-dev \
libncurses5-dev libgdbm-dev libnss3-dev \
libssl-dev libreadline-dev libffi-dev \
libsqlite3-dev wget libbz2-dev -y

# RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0rc2.tgz
# RUN tar -xf Python-3.10.*.tgz && rm -r Python-3.10.*.tgz
# RUN cd Python-3.10.0rc2 && ./configure --enable-optimizations
RUN python3 --version
RUN pip3 install torch