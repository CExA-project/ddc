# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

FROM ubuntu:noble@sha256:186072bba1b2f436cbb91ef2567abca677337cfc786c86e107d25b7072feef0c

LABEL "org.opencontainers.image.source"="https://github.com/CExA-project/ddc"

ARG BACKEND
ARG AMDGPU_VERSION=6.4.3
ARG CUDA_MAJOR_VERSION=12
ARG CUDA_MINOR_VERSION=9
ARG LLVM_VERSION=22
ARG ROCM_VERSION=6.4.3

COPY bash_run.sh /bin/
ENV BASH_ENV=/etc/profile
SHELL ["/bin/bash", "-c"]

ENV PATH="/usr/local/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}/bin${PATH:+:${PATH}}"
ENV PATH="${PATH:+$PATH:}/opt/rocm-${ROCM_VERSION}/bin"
ENV CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}/opt/rocm-${ROCM_VERSION}"

RUN chmod +x /bin/bash_run.sh \
 && export DEBIAN_FRONTEND=noninteractive \
 && echo tzdata tzdata/Areas string Etc | debconf-set-selections \
 && echo tzdata tzdata/Zones/Etc string UTC | debconf-set-selections \
 && apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    gpg \
    wget \
 && mkdir --parents --mode=0755 /etc/apt/keyrings \
 && wget --output-document=/usr/share/keyrings/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/apt.llvm.org.asc] http://apt.llvm.org/noble/ llvm-toolchain-noble-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list \
 && case "x${BACKEND}" in \
    "xhip") \
      echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/${AMDGPU_VERSION}/ubuntu noble main" > /etc/apt/sources.list.d/amdgpu.list \
      && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" > /etc/apt/sources.list.d/rocm.list \
      && wget --quiet --output-document=- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --output /etc/apt/keyrings/rocm.gpg \
      && echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600 \
 ;; "xcuda" ) \
      wget --output-document=/usr/share/keyrings/cuda-archive-keyring.asc https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
      && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cuda-archive-keyring.asc] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
      && wget --output-document=/etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin \
 ;; esac \
 && apt-get update --yes \
 && case "${BACKEND}" in \
    "cpu") \
      apt-get install --yes --no-install-recommends \
      clang-${LLVM_VERSION} \
      clang-tidy-${LLVM_VERSION} \
      && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100 \
      && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 \
      && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-${LLVM_VERSION} 100 \
 ;; "cuda") \
      # libnvjitlink seems to be a missing dependency of cusparse
      apt-get install --yes --no-install-recommends \
      cuda-minimal-build-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      cuda-nvtx-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      libcublas-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      libcufft-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      libcurand-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      libcusolver-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      libcusparse-dev-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      libnvjitlink-${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION} \
      gcc-12 \
      gfortran-12 \
      g++-12 \
 ;; "hip") \
      apt-get install --yes --no-install-recommends \
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
 && apt-get purge --yes \
    apt-transport-https \
    apt-utils \
    ca-certificates \
    wget \
 && apt-get autoremove --yes \
 && apt-get clean --yes \
 && apt-get autoclean --yes \
 && rm --force --recursive /var/lib/apt/lists/* \
 && useradd --create-home --home-dir /data --user-group ci \
 && if [ "xcuda" = "x${BACKEND}" ] \
  ; then echo 'CUDA_GCC=gcc-12' > /etc/profile.d/ddc-cuda.sh \
       ; echo 'CUDA_GXX=g++-12' >> /etc/profile.d/ddc-cuda.sh \
 ;  fi

RUN apt-get update --yes \
 && apt-get install --yes --no-install-recommends \
    file \
    bzip2 \
    ca-certificates \
    g++ \
    gcc \
    gfortran \
    git \
    gzip \
    lsb-release \
    patch \
    python3 \
    tar \
    unzip \
    xz-utils \
    zstd

USER ci:ci
WORKDIR /data

COPY spack-env /data/spack-env
RUN git clone --branch v1.2.0 --depth 1 https://github.com/spack/spack.git \
 && . /data/spack/share/spack/setup-env.sh \
 && spack env create ddc /data/spack-env/spack-${BACKEND}.yaml \
 && spack --env ddc repo update \
 && spack --env ddc install --fail-fast

ENTRYPOINT ["/bin/bash_run.sh"]
CMD ["/bin/bash", "-li"]
