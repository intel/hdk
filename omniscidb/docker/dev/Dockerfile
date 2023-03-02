FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Tools
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ccache \
    curl \
    gdb \
    git \
    libtbb-dev \
    make \
    rsync \
    tar \
    vim \
    wget \
    lsb-release \
    software-properties-common \
    --

# cmake
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libcurl4-openssl-dev \
    --

RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1.tar.gz && \
    tar xzvf cmake-3.24.1.tar.gz && \
    cd cmake-3.24.1/ && \
    ./bootstrap && \
    make -j`nproc` && \
    make install

# Toolchains
RUN apt-get update && apt-get install -y \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    libclang-14-dev \
    gcc-12 \
    g++-12 \
    openjdk-17-jdk \
    -- && \
    update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-12 800 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12

ENV PATH=/usr/lib/llvm-14/bin${PATH:+:${PATH}}

# Dependencies
RUN apt-get update && apt-get install -y \
    binutils-dev \
    bzip2 \
    libboost-all-dev \
    libbz2-dev \
    libdouble-conversion-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libncurses-dev \
    libsnappy-dev \
    libtbb-dev \
    libunwind-dev \
    libz-dev \
    maven \
    libfmt-dev \
    --

# Arrow
# https://arrow.apache.org/install/
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get update && apt-get install -y -V \
        libarrow-dev \
        libparquet-dev \
        libarrow-dataset-dev \
        libarrow-flight-dev \
        --

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean

RUN useradd -m --uid 1001 ghrunner

