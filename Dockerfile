FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Install required software and clean as not to make the layer dirty
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    apt-utils curl gnupg gcc g++ make autoconf && \
    apt-get clean && apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    git zlib1g-dev libbz2-dev liblzma-dev libzip-dev && \
    apt-get clean && apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Clone xSqueezeIt and pbwt repositories and build
RUN mkdir -p /usr/src/ && \
    cd /usr/src/ && \
    git clone https://github.com/rwk-unil/xSqueezeIt.git && \
    cd /usr/src/xSqueezeIt && \
    git submodule update --init --recursive htslib && \
    cd htslib && \
    autoheader && \
    autoconf && \
    automake --add-missing 2>/dev/null ; \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    git clone https://github.com/facebook/zstd.git && \
    cd zstd && \
    make && \
    cd .. && \
    make && \
    chmod +x xsqueezeit && \
    cp xsqueezeit /usr/local/bin/
RUN cd /usr/src/ && \
    git clone git@github.com:selfdecode/pbwt.git && \
    cd pbwt && \
    make && \
    cp ./pbwt /tool/ && \
    cd /usr/src/ && \
    rm -r xSqueezeIt

RUN apt update
RUN apt install -y bcftools tabix

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY . /tool/

CMD echo "Run with the following command : docker run <tag> python3 /tool/selphi.py [args]"
