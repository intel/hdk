FROM  --platform=linux/amd64 ubuntu:20.04

RUN apt-get update
RUN apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# HDK Deps

RUN apt-get install -y \
  build-essential \
  cmake \
  cmake-curses-gui \
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

<<<<<<< HEAD
# convenient utilities
RUN apt-get install -y zsh curl git vim
=======
# convenient utilities
RUN apt-get install -y curl git vim
>>>>>>> 656903c (Additional changes for ubuntu 20.04)

# Apache Arrow
RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V libarrow-dev libparquet-dev

# LLVM
ARG LLVM_SHA='1f9140064dfbfb0bbda8e51306ea51080b2f7aac' # 14.0.3
RUN mkdir -p /usr/local/src
WORKDIR /usr/local/src
RUN git clone https://github.com/llvm/llvm-project.git
WORKDIR /usr/local/src/llvm-project
RUN git checkout ${LLVM_SHA}
RUN cmake ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DCMAKE_INSTALL_PREFIX=../mlir-llvm \
  -DLLVM_USE_INTEL_JITEVENTS=on
RUN make -j$(nproc --all) all
RUN cmake --install .

# Gflags (for Glog)
RUN apt install -y libgflags-dev

# Unwind (glog)
RUN apt install -y libunwind-dev

# Boost
RUN apt install -y libboost-all-dev

# convenient utilities
RUN apt-get install -y curl git vim

RUN apt-get upgrade -y

RUN apt-get clean

# Gtest
RUN apt install -y libgtest-dev

# Python
RUN apt install -y python3-pip
ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt 
RUN rm /requirements.txt

# symlink pyarrow, dropping the library suffix 
RUN python3 -c "import pyarrow;print(pyarrow.create_library_symlinks())"

RUN apt-get clean
