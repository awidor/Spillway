#!/usr/bin/env bash
# Builds the S1 kit inside the kit/Dockerfile image. Run from Windows (Docker Desktop):
#   docker build -t strata-build kit
#   docker run --rm -v <repo>:/src -v strata-work:/work strata-build bash /src/kit/build.sh
# Output: /src/out/s1kit (+ s1kit.tar.gz). /work caches the llama.cpp checkout and build.
set -euo pipefail
LLAMA_SHA=781dbc5ac98921dbdb5e5b2ec5b7a50960e937d4   # master 10 Oct 2026; has --moe-cache-mib (#29887)
ARCHS="75-real;86-real"
J=$(nproc)

L=/work/llama.cpp
[ -d $L/.git ] || git clone --filter=blob:none https://github.com/ggml-org/llama.cpp $L
git -C $L fetch -q origin $LLAMA_SHA && git -C $L checkout -q $LLAMA_SHA

# Static llama/ggml libs, static libstdc++, no OpenMP: the binaries need only glibc and the bundled CUDA libs.
cmake -S $L -B /work/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_OPENMP=OFF -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
cmake --build /work/build -j $J --target llama-batched-bench

K=/src/out/s1kit
rm -rf $K && mkdir -p $K/bin $K/lib
cp /work/build/bin/llama-batched-bench $K/bin/

A=$(find /work/build -name libggml-cpu.a | head -1); B=$(find /work/build -name libggml-base.a | head -1)
nvcc -O3 -std=c++17 -ccbin g++-13 \
  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 \
  -Xcompiler "-mavx2 -mfma -mf16c -pthread -static-libstdc++ -static-libgcc" \
  -I $L/ggml/include /src/kit/src/scout.cu "$A" "$B" -o $K/bin/scout -ldl -lm

# Bundle the CUDA libraries the binaries load, and point them at ../lib.
for so in $(ldd $K/bin/* | awk '/\/usr\/local\/cuda/ {print $3}' | sort -u); do cp -L "$so" $K/lib/; done
patchelf --set-rpath '$ORIGIN/../lib' $K/bin/*

cp /src/kit/s1.sh /src/perfmodel/probe.sh $K/
sed -i 's/\r$//' $K/*.sh   # files edited on Windows may carry CRLF, which bash rejects
{ echo "llama.cpp $LLAMA_SHA"; nvcc --version | tail -1; g++-13 --version | head -1; date -u; } > $K/VERSIONS
echo "== ldd"; ldd $K/bin/*
# Everything but the driver's libcuda must be glibc or bundled: the box has nothing else.
if ldd $K/bin/* | grep -v libcuda.so | grep -E "not found|=> /(usr/)?lib.*/lib(stdc|gomp|nccl)"; then echo "unbundled dependency"; exit 1; fi
ls -la $K $K/bin $K/lib
