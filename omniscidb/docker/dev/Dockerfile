FROM ubuntu:20.04

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

# Folly
RUN apt-get install -y autoconf automake cmake libboost-all-dev libdouble-conversion-dev libevent-dev libgflags-dev libgmock-dev libgoogle-glog-dev libgtest-dev liblz4-dev liblzma-dev libsnappy-dev libsodium-dev libtool libzstd-dev ninja-build zip unzip
RUN wget https://github.com/fmtlib/fmt/releases/download/9.1.0/fmt-9.1.0.zip && \
    unzip fmt-9.1.0.zip && \
    cd fmt-9.1.0 && \
    mkdir _build && \
    cd _build && \
    cmake -DCMAKE_CXX_FLAGS="-fPIC" -DFMT_DOC=OFF -DFMT_TEST=OFF .. && \
    make -j`nproc` && \
    make install

RUN wget https://github.com/facebook/folly/releases/download/v2022.11.07.00/folly-v2022.11.07.00.tar.gz && \
    mkdir folly-2022.11.07.00 && \
    tar xzvf folly-v2022.11.07.00.tar.gz -C folly-2022.11.07.00 && \
    cd folly-2022.11.07.00 && \
    mkdir _build && \
    cd _build && \
    cmake .. && \
    make -j`nproc` && \
    make install

# Toolchains
RUN apt-get update && apt-get install -y \
    llvm-12 \
    llvm-12-dev \
    clang-12 \
    libclang-12-dev \
    gcc-9 \
    g++-9 \
    openjdk-11-jdk \
    -- && \
    update-alternatives \
        --install /usr/bin/gcc gcc /usr/bin/gcc-9 800 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9

ENV PATH=/usr/lib/llvm-12/bin${PATH:+:${PATH}}

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

