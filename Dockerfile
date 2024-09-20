FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install required software and clean as not to make the layer dirty
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    apt-utils curl gnupg gcc g++ make autoconf git zlib1g-dev libbz2-dev \
    liblzma-dev libzip-dev libcurl4-openssl-dev build-essential checkinstall && \
    apt-get clean && apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install latest bcftools and htslib
RUN mkdir -p /usr/src/ && cd /usr/src/ && \
    git clone --recurse-submodules https://github.com/samtools/htslib.git && \
    git clone https://github.com/samtools/bcftools.git && \
    cd bcftools && autoheader && autoconf && ./configure && make && make install && \
    cd /usr/src/htslib && autoreconf -i && ./configure --prefix=/usr/local && \
    make && make install && rm -fr /usr/src/bcftools /tmp/* /var/tmp/*

# Clone xSqueezeIt and build if using xsi reference panel files
RUN cd /usr/src/ && \
    git clone https://github.com/rwk-unil/xSqueezeIt.git && \
    cd /usr/src/xSqueezeIt && \
    git submodule update --init --recursive htslib && \
    cd htslib && autoheader && autoconf && \
    automake --add-missing 2>/dev/null ; ./configure && \
    make && make install && ldconfig && cd .. && \
    git clone https://github.com/facebook/zstd.git && \
    cd zstd && make && cd .. && make && chmod +x xsqueezeit && \
    cp xsqueezeit /usr/local/bin/ && \
    rm -fr /usr/src/xSqueezeIt /tmp/* /var/tmp/*

# Build pbwt library
RUN mkdir -p /tool
COPY ./pbwt /tool/pbwt
RUN cd /tool/pbwt && HTSDIR=/usr/src/htslib make

# Install python3.10 and pip
RUN apt update
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python3 python3-distutils python3-dev python3-pip
# Install requirements
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY *.py /tool/
COPY *.toml /tool/
COPY ./modules /tool/modules

ENTRYPOINT [ "python3", "/tool/selphi.py" ]
CMD [ "-h" ]
