// Decode attention timing on the proxy: Qwen3.6's geometry (16 query heads, 2 KV heads, head_dim 256, int8 K/V,
// dense selection), n_q queries of one verify window at a given context. Times the chunk and merge kernels apart.
//   nvcc -O3 -std=c++17 -arch=sm_86 -I engine/include perfmodel/attn_bench.cu -o attn_bench && ./attn_bench 30000 3
#include "../engine/src/kernels/cuda/qsa_decode_attn.cu"
#include <vector>
#include <random>
#include <cmath>
#include <cmath>
using namespace strata::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 30000, NQ = argc > 2 ? std::atoi(argv[2]) : 3;
    const int64_t cap = std::max(32768, N) + 64;
    QsaShapes s; s.n_head = 16; s.n_head_kv = 2; s.head_dim = 256; s.page_size = 4;
    const int KVH = 2, PS = 4, pages = (int) (cap / PS);
    const size_t rows = (size_t) pages * KVH * PS;
    std::mt19937 rng(1);
    std::vector<int8_t> kq(rows * 256), vq(rows * 256);
    for (auto& x : kq) x = (int8_t) (rng() % 255 - 127);
    for (auto& x : vq) x = (int8_t) (rng() % 255 - 127);
    std::vector<uint16_t> sc(rows * 4);
    for (auto& x : sc) { __half h = __float2half(0.001f + (rng() % 1000) * 1e-6f); x = __half_as_ushort(h); }
    std::vector<int32_t> pt(pages), ids((size_t) NQ * cap), steps((size_t) NQ * kStepCount, 0);
    for (int i = 0; i < pages; ++i) pt[i] = i;
    for (int q = 0; q < NQ; ++q) {
        for (int i = 0; i < N + q; ++i) ids[(size_t) q * cap + i] = i;
        steps[(size_t) q * kStepCount + kStepWidth] = N + q;
    }
    std::vector<float> qv((size_t) NQ * 16 * 256);
    for (auto& x : qv) x = (float) ((int) (rng() % 2001) - 1000) * 1e-3f;
    int8_t *dk, *dv; uint16_t *dks, *dvs; int32_t *dpt, *dids, *dst; float *dq, *dscr, *dout;
    CK(cudaMalloc(&dk, kq.size())); CK(cudaMalloc(&dv, vq.size()));
    CK(cudaMalloc(&dks, sc.size() * 2)); CK(cudaMalloc(&dvs, sc.size() * 2));
    CK(cudaMalloc(&dpt, pt.size() * 4)); CK(cudaMalloc(&dids, ids.size() * 4)); CK(cudaMalloc(&dst, steps.size() * 4));
    CK(cudaMalloc(&dq, qv.size() * 4));
    const uint64_t sf = qsa_decode_attn_scratch_floats(cap, s);
    CK(cudaMalloc(&dscr, sf * NQ * 4)); CK(cudaMalloc(&dout, qv.size() * 4));
    CK(cudaMemcpy(dk, kq.data(), kq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vq.data(), vq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dks, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvs, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpt, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dst, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dq, qv.data(), qv.size() * 4, cudaMemcpyHostToDevice));
    QsaAttnPools p; p.k_q = dk; p.v_q = dv; p.k_scale = dks; p.v_scale = dvs; p.page_table = dpt;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int R = 50;
    auto t = [&](auto fn) { fn(); cudaEventRecord(a); for (int r = 0; r < R; ++r) fn(); cudaEventRecord(b);
                            cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000 / R; };
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* pa = dscr; float* pm = dscr + (size_t) n_chunks * 16 * 256; float* pl = pm + (size_t) n_chunks * 16;
    const float scale = 1.0f / 16.0f;
    const double us_all = t([&] { qsa_decode_attn_batch(dq, p, dids, dst, cap, s, dscr, dout, NQ, nullptr); });
    const double us_chunk = t([&] { attn_chunk_kernel<1, false, 8><<<dim3(n_chunks, 2, NQ), THREADS>>>(dq, p, dids, dst, 2, PS, scale, pa, pm, pl, n_chunks, (int) cap, (long long) sf); });
    const double us_merge = t([&] { attn_merge_kernel<8><<<dim3(16, NQ), 256>>>(pa, pm, pl, n_chunks, dout, (long long) sf); });
    CK(cudaGetLastError());
    std::vector<float> ref(qv.size()), out(qv.size()), one(16 * 256);
    qsa_decode_attn_batch(dq, p, dids, dst, cap, s, dscr, dout, NQ, nullptr);
    CK(cudaMemcpy(ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost));
    const double us_dense = t([&] { qsa_decode_attn_dense(dq, p, dst, cap, s, dscr, dout, NQ, nullptr); });
    CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    double err = 0, mag = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        err = std::max(err, (double) std::fabs(out[i] - ref[i]));
        mag = std::max(mag, (double) std::fabs(ref[i]));
    }
    // each query alone (n_q = 1) must give its row of the window bit for bit
    bool same = true;
    for (int qi = 0; qi < NQ; ++qi) {
        qsa_decode_attn_dense(dq + (size_t) qi * 16 * 256, p, dst + (size_t) qi * kStepCount, cap, s, dscr, dout, 1, nullptr);
        CK(cudaMemcpy(one.data(), dout, one.size() * 4, cudaMemcpyDeviceToHost));
        same = same && std::memcmp(one.data(), out.data() + (size_t) qi * 16 * 256, one.size() * 4) == 0;
    }
    const int ns = std::min(n_chunks, 68);
    const double us_dmerge = t([&] { attn_merge_dense_kernel<<<dim3(16, NQ), 256>>>(pa, pm, pl, ns, dout, (long long) sf); });
    std::printf("dense merge alone (%d splits): %.1f us; ", ns, us_dmerge);
    const double mb = (double) N * KVH * 256 * 2 * (1 + 4.0 / 256) / 1e6;   // K+V codes + scales, read once
    std::printf("N=%d nq=%d: batch %.1f us (chunk %.1f, merge %.1f); dense %.1f us (%.0f GB/s of the K/V), max err "
                "%.1e of %.1e, alone == in window: %s\n", N, NQ, us_all, us_chunk, us_merge, us_dense,
                mb / us_dense * 1e3, err, mag, same ? "yes" : "NO");
    return err <= 1e-4 * mag && same ? 0 : 1;
}
