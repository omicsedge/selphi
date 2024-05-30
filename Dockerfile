FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Install required software and clean as not to make the layer dirty
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    apt-utils curl gnupg gcc g++ make autoconf git zlib1g-dev libbz2-dev \
    liblzma-dev libzip-dev libcurl4-openssl-dev bcftools tabix && \
    apt-get clean && apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install python and pip
RUN apt update
RUN apt install -y build-essential checkinstall
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3.8 python3.8-distutils python3.8-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py

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
# Build pbwt library
RUN mkdir -p /tool
RUN apt install -y libssl-dev gdb
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
COPY ./pbwt /tool/pbwt
RUN cd /tool/pbwt && HTSDIR=/usr/src/xSqueezeIt/htslib make
# clean up
RUN rm -r /usr/src/xSqueezeIt



COPY *.py /tool/
COPY *.toml /tool/
COPY ./modules /tool/modules

ENTRYPOINT [ "python3", "/tool/selphi.py" ]
CMD [ "-h" ]
