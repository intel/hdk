FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Tools
RUN apt-get update && apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ccache \
    cmake \
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

# Toolchains
RUN apt-get update && apt-get install -y \
    gcc-9 \
    g++-9 \
    openjdk-11-jdk \
    -- && \
    update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-9 800 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9

# LLVM
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository \
        'deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main' && \
    apt-get update && apt-get install -y \
        llvm-14 \
        llvm-14-dev \
        clang-14 \
        libclang-14-dev \
        --

ENV PATH=/usr/lib/llvm-14/bin${PATH:+:${PATH}}

# Dependencies
RUN apt-get update && apt-get install -y \
    binutils-dev \
    bzip2 \
    libboost-all-dev \
    libbz2-dev \
    libdouble-conversion-dev \
    libgflags-dev \
    libgtest-dev \
    libncurses-dev \
    libsnappy-dev \
    libtbb-dev \
    libunwind-dev \
    libz-dev \
    maven \
    libfmt-dev \
    cython3 \
    python3-numpy \
    python3-pip \
    sqlite3 \
    libsqlite3-dev \
    --

# Arrow
# https://arrow.apache.org/install/
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get update && apt-get install -y -V \
        libarrow-dev=11* \
        libparquet-dev=11* \
        libarrow-dataset-dev=11* \
        libarrow-flight-dev=11* \
        --

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean

RUN useradd -m --uid 1002 ghrunner

