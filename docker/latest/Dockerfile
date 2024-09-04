# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble

LABEL "org.opencontainers.image.source"="https://github.com/Maison-de-la-Simulation/ddc"

ARG BACKEND
ARG AMDGPU_VERSION=6.2
ARG ROCM_VERSION=6.2

COPY bash_run /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PATH="${PATH:+$PATH:}/opt/rocm-${ROCM_VERSION}.0/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}/opt/rocm-${ROCM_VERSION}.0"

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
 && if [ "xhip" = "x${BACKEND}" ] \
  ; then echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/ubuntu noble main" > /etc/apt/sources.list.d/amdgpu.list \
      && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" > /etc/apt/sources.list.d/rocm.list \
      && wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --output /etc/apt/keyrings/rocm.gpg \
      && echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600 \
 ;  fi \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/pdidev-archive-keyring.gpg] https://raw.githubusercontent.com/pdidev/repo/ubuntu noble main" > /etc/apt/sources.list.d/pdi.list \
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
    pkg-config \
 && case "${BACKEND}" in \
    "cpu") \
      apt-get install -y --no-install-recommends \
      clang \
      clang-tidy \
      clang-format \
 ;; "cuda") \
      apt-get install -y --no-install-recommends \
      nvidia-cuda-toolkit \
      gcc-12 \
      g++-12 \
 ;; "hip") \
      apt-get install -y --no-install-recommends \
      rocm-hip-sdk \
      && echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
      && echo "/opt/rocm/lib64" >> /etc/ld.so.conf.d/rocm.conf \
      && ldconfig \
;; esac \
 && git clone -b v1.8.0 https://github.com/ginkgo-project/ginkgo.git \
 && cd ginkgo \
 && case "${BACKEND}" in \
    "cpu") \
      cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGINKGO_BUILD_OMP=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
 ;; "cuda") \
      cmake -S . -B build -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGINKGO_CUDA_ARCHITECTURES=70 -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
 ;; "hip") \
      # HIP_PATH needs to be set to avoid Ginkgo defining it (https://github.com/ginkgo-project/ginkgo/blob/develop/cmake/hip_path.cmake).
      # The Ginkgo heuristic does not seem to be compatible with the new HIP v6 directory layout.
      # This makes amdclang++ fail during cmake compiler test (https://releases.llvm.org/18.1.0/tools/clang/docs/HIPSupport.html#order-of-precedence-for-hip-path)
      export HIP_PATH=`hipconfig --path` \
      && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a -DGINKGO_BUILD_HIP=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
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
  ; then echo 'CUDA_GCC=gcc-12' > /etc/profile.d/ddc-cuda.sh \
       ; echo 'CUDA_GXX=g++-12' >> /etc/profile.d/ddc-cuda.sh \
 ;  fi

USER ci:ci
WORKDIR /data

ENTRYPOINT ["/bin/bash_run"]
CMD ["/bin/bash", "-li"]


