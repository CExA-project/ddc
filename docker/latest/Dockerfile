# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble

LABEL "org.opencontainers.image.source"="https://github.com/CExA-project/ddc"

ARG BACKEND
ARG AMDGPU_VERSION=6.2.1
ARG ROCM_VERSION=6.2.1

COPY bash_run /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PATH="/usr/local/cuda-12.6/bin${PATH:+:${PATH}}"
ENV PATH="/opt/cmake-3.27.9-linux-x86_64/bin${PATH:+:${PATH}}"
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
 && wget -O /usr/share/keyrings/pdidev-archive-keyring.gpg https://repo.pdi.dev/ubuntu/pdidev-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pdidev-archive-keyring.gpg] https://repo.pdi.dev/ubuntu noble main" > /etc/apt/sources.list.d/pdi.list \
 && case "x${BACKEND}" in \
    "xhip") \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/ubuntu noble main" > /etc/apt/sources.list.d/amdgpu.list \
      && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" > /etc/apt/sources.list.d/rocm.list \
      && wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --output /etc/apt/keyrings/rocm.gpg \
      && echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600 \
 ;; "xcuda" ) \
      wget -O /usr/share/keyrings/cuda-archive-keyring.asc https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
      && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cuda-archive-keyring.asc] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
      && wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin \
 ;; esac \
 && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libfftw3-dev \
    libhwloc-dev \
    liblapacke-dev \
    libpdi-dev \
    pdiplugin-decl-hdf5-serial \
    pdiplugin-trace \
    pdiplugin-user-code \
    pkg-config \
    pipx \
    python3 \
 && python3 -m pipx install gcovr \
 && python3 -m pipx ensurepath \
 # Installing cmake < 3.28 to workaround issue with Kokkos
 && wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.tar.gz \
 && tar -xvf cmake-3.27.9-linux-x86_64.tar.gz --directory /opt \
 && rm -f cmake-3.27.9-linux-x86_64.tar.gz \
 && case "${BACKEND}" in \
    "cpu") \
      apt-get install -y --no-install-recommends \
      clang \
      clang-tidy \
      clang-format \
 ;; "cuda") \
      # libnvjitlink-12-6 seems to be a missing dependency of cusparse
      apt-get install -y --no-install-recommends \
      cuda-minimal-build-12-6 \
      cuda-nvtx-12-6 \
      libcublas-dev-12-6 \
      libcufft-dev-12-6 \
      libcurand-dev-12-6 \
      libcusparse-dev-12-6 \
      libnvjitlink-12-6 \
      gcc-12 \
      g++-12 \
 ;; "hip") \
      apt-get install -y --no-install-recommends \
      # libxml2 seems to be a missing dependency of rocm-hip-runtime
      libxml2 \
      hipblas-dev${ROCM_VERSION} \
      hipfft-dev${ROCM_VERSION} \
      hiprand-dev${ROCM_VERSION} \
      hipsparse-dev${ROCM_VERSION} \
      rocm-hip-runtime-dev${ROCM_VERSION} \
      rocrand-dev${ROCM_VERSION} \
      rocthrust-dev${ROCM_VERSION} \
      && echo "/opt/rocm/lib" >> /etc/ld.so.conf.d/rocm.conf \
      && echo "/opt/rocm/lib64" >> /etc/ld.so.conf.d/rocm.conf \
      && ldconfig \
 ;; esac \
 && git clone -b v1.8.0 --depth 1 https://github.com/ginkgo-project/ginkgo.git \
 && case "${BACKEND}" in \
    "cpu") \
      cmake -DCMAKE_BUILD_TYPE=Release -DGINKGO_BUILD_OMP=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -B ginkgo/build -S ginkgo \
 ;; "cuda") \
      cmake -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12 -DCMAKE_BUILD_TYPE=Release -DGINKGO_CUDA_ARCHITECTURES=70 -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -B ginkgo/build -S ginkgo \
 ;; "hip") \
      # HIP_PATH needs to be set to avoid Ginkgo defining it (https://github.com/ginkgo-project/ginkgo/blob/develop/cmake/hip_path.cmake).
      # The Ginkgo heuristic does not seem to be compatible with the new HIP v6 directory layout.
      # This makes amdclang++ fail during cmake compiler test (https://releases.llvm.org/18.1.0/tools/clang/docs/HIPSupport.html#order-of-precedence-for-hip-path)
      export HIP_PATH=`hipconfig --path` \
      && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a -DGINKGO_BUILD_HIP=ON -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_EXAMPLES=OFF -DGINKGO_BUILD_BENCHMARKS=OFF -B ginkgo/build -S ginkgo \
 ;; esac \
 && cmake --build ginkgo/build \
 && cmake --install ginkgo/build --prefix /usr \
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


