FROM ubuntu:20.04

RUN apt-get update
RUN apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# HDK Deps

RUN apt-get install -y \
  build-essential \
  cmake \
  gdb \
  make \
  autoconf \
  automake \
  rsync \
  tar \
  wget \
  python \
  python-dev 

# Apache Arrow
RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V libarrow-dev 

# LLVM
RUN apt install -y llvm-12 llvm-12-dev

# Gflags (for Glog)
RUN apt install -y libgflags-dev

# Boost
RUN apt install -y libboost-all-dev

# GDAL
RUN apt-get install -y libgdal-dev

# convenient utilities
RUN apt-get install -y zsh curl git vim

RUN apt-get upgrade -y

RUN apt-get clean

# Gtest
RUN apt install -y libgtest-dev

# Unwind (glog)
RUN apt install -y libunwind-dev

# libarchive
RUN apt install -y libarchive-dev

RUN apt-get clean
