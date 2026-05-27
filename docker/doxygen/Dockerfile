# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble@sha256:186072bba1b2f436cbb91ef2567abca677337cfc786c86e107d25b7072feef0c

LABEL "org.opencontainers.image.source"="https://github.com/CExA-project/ddc"

COPY bash_run.sh /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

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
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    clang \
    cm-super-minimal \
    doxygen-latex \
    ghostscript \
    graphviz \
 && apt-get purge -y \
    apt-transport-https \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && apt-get autoclean -y \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -d /data -m -U ci

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends file bzip2 ca-certificates g++ gcc gfortran git gzip lsb-release patch python3 tar unzip xz-utils zstd

USER ci:ci
WORKDIR /data

COPY spack-env /data/spack-env
RUN git clone --branch v1.1.1 --depth 1 https://github.com/spack/spack.git \
 && . /data/spack/share/spack/setup-env.sh \
 && spack env create ddc /data/spack-env/spack-cpu.yaml \
 && spack --env ddc repo update \
 && spack --env ddc install --fail-fast

ENTRYPOINT ["/bin/bash_run.sh"]
CMD ["/bin/bash", "-li"]
