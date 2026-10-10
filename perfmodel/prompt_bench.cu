// Prompt attention timing on the proxy: Qwen3.6's geometry (16 query heads, 2 KV heads, head_dim 256, int8 K/V,
// dense causal), one prompt chunk of T positions ending at `ctx`. Times qsa_prompt_attn_batch (what prefill runs;
// STRATA_EMULATE_CC=75 takes Turing's kernel) and checks it against the FP32 decode kernel.
//   nvcc -O3 -std=c++17 -arch=sm_86 -I engine/include perfmodel/prompt_bench.cu \
//        engine/src/kernels/cuda/qsa_prompt_attn.cu engine/src/kernels/cuda/qsa_decode_attn.cu -o prompt_bench
//   ./prompt_bench 30000 2048
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace strata::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); return 2; } } while (0)
int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 30000, T = argc > 2 ? std::atoi(argv[2]) : 2048;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 3;
    const int64_t cap = N + 64;
    QsaShapes s; s.n_head = 16; s.n_head_kv = 2; s.head_dim = 256; s.page_size = 4;
    const int KVH = 2, PS = 4, pages = (int) (cap / PS);
    const size_t rows = (size_t) pages * KVH * PS;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> kq(rows * 256), vq(rows * 256);
    for (auto& x : kq) x = (int8_t) (rng() % 255 - 127);
    for (auto& x : vq) x = (int8_t) (rng() % 255 - 127);
    std::vector<uint16_t> sc(rows * 4);
    for (auto& x : sc) x = __half_as_ushort(__float2half(0.002f + (rng() % 1000) * 2e-5f));
    std::vector<int32_t> pt(pages), steps((size_t) T * kStepCount, 0);
    for (int i = 0; i < pages; ++i) pt[i] = i;
    for (int i = 0; i < T; ++i) steps[(size_t) i * kStepCount + kStepWidth] = N - T + i + 1;   // causal: position + 1 cells
    std::vector<float> qv((size_t) T * 16 * 256);
    for (auto& x : qv) x = nd(rng) * 2.0f;
    int8_t *dk, *dv; uint16_t *dks, *dvs; int32_t *dpt, *dst; float *dq, *dscr, *dref, *dout;
    CK(cudaMalloc(&dk, kq.size())); CK(cudaMalloc(&dv, vq.size()));
    CK(cudaMalloc(&dks, sc.size() * 2)); CK(cudaMalloc(&dvs, sc.size() * 2));
    CK(cudaMalloc(&dpt, pt.size() * 4)); CK(cudaMalloc(&dst, steps.size() * 4));
    CK(cudaMalloc(&dq, qv.size() * 4)); CK(cudaMalloc(&dref, qv.size() * 4)); CK(cudaMalloc(&dout, qv.size() * 4));
    const uint64_t sf = qsa_decode_attn_scratch_floats(cap, s);
    CK(cudaMalloc(&dscr, sf * 64 * 4));
    CK(cudaMemcpy(dk, kq.data(), kq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vq.data(), vq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvs, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpt, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dst, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice));
    QsaAttnPools p; p.k_q = dk; p.v_q = dv; p.k_scale = dks; p.v_scale = dvs; p.page_table = dpt;
    // FP32 reference: the CUDA-core decode kernel (not its tensor-core path), 64 positions at a time
    setenv("STRATA_ATTN_TC", "0", 1);
    for (int t0 = 0; t0 < T; t0 += 64) {
        const int nb = std::min(64, T - t0);
        if (!qsa_decode_attn_dense(dq + (size_t) t0 * 16 * 256, p, dst + (size_t) t0 * kStepCount, cap, s, dscr,
                                   dref + (size_t) t0 * 16 * 256, nb, nullptr)) { std::printf("dense refused\n"); return 2; }
    }
    if (!qsa_prompt_attn_batch(dq, p, nullptr, dst, cap, s, dout, T, nullptr)) { std::printf("prompt refused\n"); return 2; }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) qsa_prompt_attn_batch(dq, p, nullptr, dst, cap, s, dout, T, nullptr);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= reps;
    CK(cudaGetLastError());
    std::vector<float> ref(qv.size()), out(qv.size());
    CK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    double err = 0, mag = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        err = std::max(err, (double) std::fabs(out[i] - ref[i]));
        mag = std::max(mag, (double) std::fabs(ref[i]));
    }
    uint64_t hash = 1469598103934665603ull;   // FNV-1a of the output bytes: equal across kernels = bit for bit
    for (const unsigned char c : std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(out.data()),
                                                             reinterpret_cast<const unsigned char*>(out.data() + out.size())))
        hash = (hash ^ c) * 1099511628211ull;
    const double tflop = 2.0 * 2.0 * 16 * 256 * ((double) T * (N - T) + 0.5 * T * T) / 1e12;   // QK + PV, causal
    std::printf("ctx %d, %d positions: %.2f ms (%.1f TFLOPS useful, %.0f positions/s per layer); max err %.2e of %.2e "
                "(%.1e relative); output %016llx\n", N, T, ms, tflop / ms * 1e3, T / ms * 1e3, err, mag, err / mag,
                (unsigned long long) hash);
    return err <= 1e-3 * mag ? 0 : 1;
}
