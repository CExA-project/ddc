FROM ubuntu:noble

# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

COPY bash_run /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

RUN chmod +x /bin/bash_run \
 && export DEBIAN_FRONTEND=noninteractive \
 && echo tzdata tzdata/Areas string Etc | debconf-set-selections \
 && echo tzdata tzdata/Zones/Etc string UTC | debconf-set-selections \
 && apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cm-super-minimal \
    cmake \
    cmake-data \
    libfftw3-dev \
    liblapacke-dev \
    ghostscript \
    git \
    graphviz \
    pkg-config \
    doxygen-latex \
 && git clone --branch 1.8.1 --depth 1 https://github.com/pdidev/pdi.git \
 && cmake -D BUILD_BENCHMARKING=OFF -D BUILD_DECL_HDF5_PLUGIN=OFF -D BUILD_FORTRAN=OFF -D BUILD_HDF5_PARALLEL=OFF -D BUILD_DECL_NETCDF_PLUGIN=OFF -D BUILD_MPI_PLUGIN=OFF -D BUILD_NETCDF_PARALLEL=OFF -D BUILD_SERIALIZE_PLUGIN=OFF -D BUILD_SET_VALUE_PLUGIN=OFF -D BUILD_SHARED_LIBS=ON -D BUILD_TESTING=OFF -D BUILD_TRACE_PLUGIN=ON -D BUILD_USER_CODE_PLUGIN=ON -D CMAKE_BUILD_TYPE=Release -D USE_paraconf=EMBEDDED -D USE_spdlog=EMBEDDED -D USE_yaml=EMBEDDED -B pdi/build -S pdi \
 && cmake --build pdi/build \
 && cmake --install pdi/build \
 && rm -rf pdi \
 && git clone --branch v1.8.0 --depth 1 https://github.com/ginkgo-project/ginkgo.git \
 && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGINKGO_BUILD_OMP=OFF -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -B ginkgo/build -S ginkgo \
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

ENTRYPOINT ["/bin/bash_run"]
CMD ["/bin/bash", "-li"]
