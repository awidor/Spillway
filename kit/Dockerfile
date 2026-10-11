# Build image for every Linux binary that goes to the laptop (docs/design/plan.md, phase 0).
# Ubuntu 20.04 base: glibc 2.31, so the binaries run on any distro the laptop may have.
# CUDA 12.8: runs on any 525+ driver; sm_75: the laptop's RTX 2070 Max-Q (Turing), the only target.
FROM nvidia/cuda:12.8.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common gpg-agent ca-certificates git \
      python3-pip ninja-build patchelf \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update \
 && apt-get install -y --no-install-recommends gcc-13 g++-13 \
 && pip3 install --no-cache-dir cmake==3.31.6 \
 && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13
