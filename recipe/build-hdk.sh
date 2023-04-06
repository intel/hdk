#!/usr/bin/env bash

set -ex

set_maven_proxy() {
    mkdir -p ~/.m2

    cat <<EOF >~/.m2/settings.xml
<settings>
  <proxies>
    <proxy>
      <active>true</active>
      <protocol>https</protocol>
      <host>$1</host>
      <port>$2</port>
    </proxy>
  </proxies>
</settings>
EOF
    cat ~/.m2/settings.xml
}

# Set Maven proxy if any
test -z "$HTTPS_PROXY" || set_maven_proxy $(echo $HTTPS_PROXY | sed -e 's#.*/##g; s#:# #')


export INSTALL_BASE=opt/hdk
test -z "$CC" || export CUDAFLAGS="--compiler-bindir $CC"

cmake -Wno-dev \
    -DCMAKE_PREFIX_PATH=$PREFIX \
    -DCMAKE_INSTALL_PREFIX=$PREFIX/$INSTALL_BASE \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_AWS_S3=off -B build -S .

cmake --build build --parallel $(nproc)
cmake --install build --prefix $PREFIX/$INSTALL_BASE

