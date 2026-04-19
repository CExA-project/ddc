# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble@sha256:186072bba1b2f436cbb91ef2567abca677337cfc786c86e107d25b7072feef0c

LABEL "org.opencontainers.image.source"="https://github.com/CExA-project/ddc"

ARG CMAKE_VERSION=4.2.3
ARG GINKGO_VERSION=1.11.0

COPY bash_run.sh /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PATH="/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin${PATH:+:${PATH}}"

RUN chmod +x /bin/bash_run.sh \
 && export DEBIAN_FRONTEND=noninteractive \
 && echo tzdata tzdata/Areas string Etc | debconf-set-selections \
 && echo tzdata tzdata/Zones/Etc string UTC | debconf-set-selections \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && wget -O /usr/share/keyrings/pdidev-archive-keyring.gpg https://repo.pdi.dev/ubuntu/pdidev-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pdidev-archive-keyring.gpg] https://repo.pdi.dev/ubuntu noble main" > /etc/apt/sources.list.d/pdi.list \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cm-super-minimal \
    doxygen-latex \
    ghostscript \
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
 && apt-get purge -y \
    apt-transport-https \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && apt-get autoclean -y \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -d /data -m -U ci

USER ci:ci
WORKDIR /data

ENTRYPOINT ["/bin/bash_run.sh"]
CMD ["/bin/bash", "-li"]
