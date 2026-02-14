# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble@sha256:c35e29c9450151419d9448b0fd75374fec4fff364a27f176fb458d472dfc9e54

LABEL "org.opencontainers.image.source"="https://github.com/CExA-project/ddc"

ARG CMAKE_VERSION=4.2.3
ARG DOXYGEN_VERSION=1.16.1
ARG GINKGO_VERSION=1.11.0
ARG LLVM_VERSION=21

COPY bash_run /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PATH="/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin${PATH:+:${PATH}}"

RUN chmod +x /bin/bash_run \
 && export DEBIAN_FRONTEND=noninteractive \
 && echo tzdata tzdata/Areas string Etc | debconf-set-selections \
 && echo tzdata tzdata/Zones/Etc string UTC | debconf-set-selections \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && wget -O /usr/share/keyrings/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key \
 && wget -O /usr/share/keyrings/pdidev-archive-keyring.gpg https://repo.pdi.dev/ubuntu/pdidev-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/apt.llvm.org.asc] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pdidev-archive-keyring.gpg] https://repo.pdi.dev/ubuntu noble main" > /etc/apt/sources.list.d/pdi.list \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    clang-${LLVM_VERSION} \
    libclang-${LLVM_VERSION}-dev \
    llvm-${LLVM_VERSION}-dev \
    cm-super-minimal \
    ghostscript \
    libc6 \
    texlive-extra-utils \
    texlive-font-utils \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-plain-generic \
    bison \
    flex \
    build-essential \
    python3 \
    git \
    graphviz \
    libfftw3-dev \
    liblapacke-dev \
    libpdi-dev \
    pkg-config \
 && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz \
 && tar -xvf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --directory /opt \
 && rm -f cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz \
 && git clone --branch v${GINKGO_VERSION} --depth 1 https://github.com/ginkgo-project/ginkgo.git \
 && cmake -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_OMP=OFF -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -B ginkgo/build -S ginkgo \
 && cmake --build ginkgo/build \
 && cmake --install ginkgo/build --prefix /usr \
 && rm -rf ginkgo \
 && wget https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.src.tar.gz \
 && tar -xvf doxygen-${DOXYGEN_VERSION}.src.tar.gz \
 && rm -f doxygen-${DOXYGEN_VERSION}.src.tar.gz \
 && cmake -D CMAKE_PREFIX_PATH=/usr/lib/llvm-21 -D CMAKE_BUILD_TYPE=Release -D use_libclang=ON -D use_sys_fmt=ON -D use_sys_spdlog=ON -B doxygen-${DOXYGEN_VERSION}/build -S doxygen-${DOXYGEN_VERSION} \
 && cmake --build doxygen-${DOXYGEN_VERSION}/build \
 && cmake --install doxygen-${DOXYGEN_VERSION}/build --prefix /usr \
 && rm -rf doxygen-${DOXYGEN_VERSION} \
 && apt-get purge -y \
    apt-transport-https \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && apt-get autoclean -y \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -d /data -m -U ci

USER ci:ci
WORKDIR /data

ENTRYPOINT ["/bin/bash_run"]
CMD ["/bin/bash", "-li"]
