// Host -> VRAM throughput of the PCIe share's paths: the copy kernel reading mapped (pinned) host memory with a few
// blocks, against the copy engine (cudaMemcpyAsync).  2 blobs of B bytes a "layer", like a decode window's misses.
//   nvcc -O3 -arch=sm_75 -o out/zc_bench perfmodel/zc_bench.cu && out/zc_bench [blob_kib]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void fetch(const uint4* __restrict__ src, uint4* __restrict__ dst, long long n) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
__global__ void fetch4(const uint4* __restrict__ src, uint4* __restrict__ dst, long long n) {   // 4 loads in flight a thread
    const long long s = (long long) gridDim.x * blockDim.x;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += 4 * s) {
        uint4 v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) if (i + k * s < n) v[k] = src[i + k * s];
#pragma unroll
        for (int k = 0; k < 4; ++k) if (i + k * s < n) dst[i + k * s] = v[k];
    }
}

int main(int argc, char** argv) {
    const size_t blob = (size_t) (argc > 1 ? atoi(argv[1]) : 640) * 1024, layer = 2 * blob, total = 64 * layer;
    uint8_t *h, *d;
    cudaHostAlloc(&h, total, cudaHostAllocMapped);
    for (size_t i = 0; i < total; i += 4096) h[i] = (uint8_t) i;
    uint8_t* hd;
    cudaHostGetDevicePointer((void**) &hd, h, 0);
    cudaMalloc(&d, layer);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    auto run = [&](const char* name, auto&& one) {
        for (int w = 0; w < 2; ++w) {
            cudaEventRecord(a);
            for (int l = 0; l < 64; ++l) one(l);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
        }
        float ms; cudaEventElapsedTime(&ms, a, b);
        std::printf("%-28s %6.2f GB/s  (%.1f us a %zu KiB layer)\n", name, total / ms / 1e6, ms * 1e3 / 64, layer / 1024);
    };
    for (int blocks : {4, 8, 16, 40, 384}) {
        char n[64];
        std::snprintf(n, sizeof n, "copy kernel, %d blocks", blocks);
        run(n, [&](int l) { fetch<<<blocks, 256>>>((const uint4*) (hd + l * layer), (uint4*) d, (long long) (layer / 16)); });
        std::snprintf(n, sizeof n, "copy kernel x4, %d blocks", blocks);
        run(n, [&](int l) { fetch4<<<blocks, 256>>>((const uint4*) (hd + l * layer), (uint4*) d, (long long) (layer / 16)); });
    }
    run("copy engine", [&](int l) { cudaMemcpyAsync(d, h + l * layer, layer, cudaMemcpyHostToDevice); });
    std::printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
