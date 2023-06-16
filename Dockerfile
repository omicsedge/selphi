FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Install required software and clean as not to make the layer dirty
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    apt-utils curl gnupg gcc g++ make autoconf && \
    apt-get clean && apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    git zlib1g-dev libbz2-dev liblzma-dev libzip-dev libcurl4-openssl-dev && \
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
# Clone and build pbwt library in same directory to share htslib
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone -b feat/merge-ref-target-files \
    git+ssh://git@github.com/selfdecode/pbwt.git /usr/src/xSqueezeIt/pbwt && \
    cd /usr/src/xSqueezeIt/pbwt && \
    make && \
    mkdir - p /tool && \
    cp ./pbwt /tool/ && \
    cd /usr/src/ && \
    rm -r xSqueezeIt

RUN apt update
RUN apt install -y bcftools tabix

RUN apt install -y make build-essential checkinstall
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3.8 python3.8-distutils python3.8-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY . /tool/

ENTRYPOINT [ "python3", "/tool/selphi.py" ]
CMD [ "-h" ]
