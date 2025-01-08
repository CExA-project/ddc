# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:jammy

LABEL "org.opencontainers.image.source"="https://github.com/Maison-de-la-Simulation/ddc"

ARG BACKEND
ARG AMDGPU_VERSION=5.7.3
ARG ROCM_VERSION=5.7.3

COPY bash_run /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PATH="/usr/local/cuda-11.8/bin${PATH:+:${PATH}}"
ENV PATH="${PATH:+$PATH:}/opt/rocm-${ROCM_VERSION}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}/opt/rocm-${ROCM_VERSION}"

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
    gpg \
    wget \
 && mkdir --parents --mode=0755 /etc/apt/keyrings \
 && case "x${BACKEND}" in \
    "xhip") \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/ubuntu jammy main" > /etc/apt/sources.list.d/amdgpu.list \
      && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" > /etc/apt/sources.list.d/rocm.list \
      && wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --output /etc/apt/keyrings/rocm.gpg \
      && echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600 \
 ;; "xcuda" ) \
      wget -O /usr/share/keyrings/cuda-archive-keyring.asc https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
      && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cuda-archive-keyring.asc] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
      && wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
 ;; esac \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/pdidev-archive-keyring.gpg] https://raw.githubusercontent.com/pdidev/repo/ubuntu jammy main" > /etc/apt/sources.list.d/pdi.list \
 && wget -q -O /etc/apt/keyrings/pdidev-archive-keyring.gpg https://raw.githubusercontent.com/pdidev/repo/ubuntu/pdidev-archive-keyring.gpg \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libfftw3-dev \
    liblapacke-dev \
    libhwloc-dev \
    libpdi-dev \
    pdidev-archive-keyring \
    pdiplugin-user-code \
    pkg-config \
 && case "${BACKEND}" in \
    "cpu") \
      apt-get install -y --no-install-recommends \
      clang \
 ;; "cuda") \
      apt-get install -y --no-install-recommends \
      cuda-minimal-build-11-8 \
      cuda-nvtx-11-8 \
      libcublas-dev-11-8 \
      libcufft-dev-11-8 \
      libcurand-dev-11-8 \
      libcusparse-dev-11-8 \
      gcc-10 \
      g++-10 \
 ;; "hip") \
      apt-get install -y --no-install-recommends \
      hipblas-dev${ROCM_VERSION} \
      hipfft-dev${ROCM_VERSION} \
      hipsparse-dev${ROCM_VERSION} \
      rocm-hip-runtime-dev${ROCM_VERSION} \
      rocrand-dev${ROCM_VERSION} \
      rocthrust-dev${ROCM_VERSION} \
      && echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
      && echo "/opt/rocm/lib64" >> /etc/ld.so.conf.d/rocm.conf \
      && ldconfig \
 ;; esac \
 && git clone -b v1.8.0 https://github.com/ginkgo-project/ginkgo.git \
 && cd ginkgo \
 && case "${BACKEND}" in \
    "cpu") \
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_OMP=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
 ;; "cuda") \
      cmake -S . -B build -DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10 -DCMAKE_CUDA_HOST_COMPILER=g++-10 -DCMAKE_BUILD_TYPE=Release -DGINKGO_CUDA_ARCHITECTURES=70 -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
 ;; "hip") \
      cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a -DGINKGO_BUILD_HIP=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
 ;; esac \
 && cmake --build build \
 && cmake --install build \
 && cd .. \
 && rm -rf ginkgo \
 && apt-get purge -y \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && apt-get autoremove -y \
 && apt-get clean -y \
 && apt-get autoclean -y \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -d /data -m -U ci \
 && if [ "xcuda" = "x${BACKEND}" ] \
  ; then echo 'CUDA_GCC=gcc-10' > /etc/profile.d/ddc-cuda.sh \
       ; echo 'CUDA_GXX=g++-10' >> /etc/profile.d/ddc-cuda.sh \
 ;  fi

USER ci:ci
WORKDIR /data

ENTRYPOINT ["/bin/bash_run"]
CMD ["/bin/bash", "-li"]


