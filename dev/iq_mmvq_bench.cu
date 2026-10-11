// Dense decode GEMVs at Qwen3.6-35B-A3B's shapes (IQ3_XXS / IQ4_XS weights, T = 1..5 activation columns): time per
// call and the share of VRAM bandwidth it reaches, and an FNV hash of the outputs (a kernel change must keep it).
//   nvcc -O3 -std=c++17 -arch=sm_75 -Iengine/include -o out/iq_mmvq_bench dev/iq_mmvq_bench.cu \
//        /work/engine-build/libstrata_kernels.a && out/iq_mmvq_bench
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

int main(int argc, char** argv) {
    struct Shape { const char* name; int type, n_in, n_out, block_bytes; bool native; };
    const Shape shapes[] = {
        {"GDN qkv  IQ3_XXS", 18, 2048, 8192, 98, false}, {"GDN qkv  IQ4_XS ", 23, 2048, 8192, 136, true},
        {"GDN z    IQ3_XXS", 18, 2048, 4096, 98, false}, {"GDN out  IQ3_XXS", 18, 4096, 2048, 98, false},
        {"QSA q    IQ3_XXS", 18, 2048, 9216, 98, false}, {"GDN out  Q5_K   ", 13, 4096, 2048, 176, true},
    };
    const int only = argc > 1 ? atoi(argv[1]) : -1;
    std::mt19937 rng(3);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (int si = 0; si < (int) (sizeof shapes / sizeof shapes[0]); ++si) {
        if (only >= 0 && si != only) continue;
        const Shape& s = shapes[si];
        const size_t wb = (size_t) s.n_out * (s.n_in / 256) * s.block_bytes;
        std::vector<uint8_t> w(wb);
        for (auto& v : w) v = (uint8_t) rng();
        for (size_t i = 0; i < wb; i += s.block_bytes) {   // a sane fp16 scale d at the block's start
            const uint16_t h = (uint16_t) (((rng() & 1) << 15) | ((5u + rng() % 5u) << 10) | (rng() & 0x3ffu));
            std::memcpy(&w[i], &h, 2);
        }
        std::vector<float> x((size_t) 5 * s.n_in);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        uint8_t *dw, *dq; float *dx, *dy;
        cudaMalloc(&dw, wb); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy, (size_t) 5 * s.n_out * 4);
        cudaMalloc(&dq, k::native_q8_1_bytes(s.n_in, 5));
        cudaMemcpy(dw, w.data(), wb, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        for (int T = 1; T <= 5; ++T) {
            if (s.native) k::native_quantize_q8_1(dx, dq, s.n_in, T, st);
            else k::quantize_q8_1_rows(dx, T, s.n_in, dq, st);
            auto call = [&] {   // the engine's path: native_mmvq for the k-quants and IQ4_XS, iq_mmvq for the i-quants
                if (s.native) k::native_mmvq(s.type, dw, dq, dy, s.n_in, s.n_out, T, st);
                else k::iq_mmvq(s.type, dw, dq, dy, s.n_in, s.n_out, T, st);
            };
            for (int r = 0; r < 3; ++r) call();
            const int reps = 200;
            cudaEventRecord(a, st);
            for (int r = 0; r < reps; ++r) call();
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            const double us = 1000.0 * ms / reps;
            std::vector<float> y((size_t) T * s.n_out);
            cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost);
            uint64_t h = 1469598103934665603ull;
            for (float v : y) { uint32_t u; std::memcpy(&u, &v, 4); h = (h ^ u) * 1099511628211ull; }
            std::printf("%s  %.2f MB  T %d  %7.2f us  %6.1f GB/s  hash %016llx\n", s.name, wb / 1e6, T, us, wb / us / 1e3,
                        (unsigned long long) h);
        }
        cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dq);
    }
    std::printf("%s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}
