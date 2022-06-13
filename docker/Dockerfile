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
  python-dev \
  libtbb-dev \
  openjdk-11-jdk 

# Maven (after openjdk)
RUN apt-get install -y maven 

# Apache Arrow
RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V libarrow-dev libparquet-dev

# LLVM
RUN apt install -y llvm-12 llvm-12-dev clang-12

# Gflags (for Glog)
RUN apt install -y libgflags-dev

# Boost
RUN apt install -y libboost-all-dev

# convenient utilities
RUN apt-get install -y curl git vim

RUN apt-get upgrade -y

RUN apt-get clean

# Gtest
RUN apt install -y libgtest-dev

# Unwind (glog)
RUN apt install -y libunwind-dev

# Python
RUN apt install -y python3-pip
ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt 
RUN rm /requirements.txt

# symlink pyarrow, dropping the library suffix 
RUN python3 -c "import pyarrow;print(pyarrow.create_library_symlinks())"

RUN apt-get clean
