// scout: the scouting micro-benchmarks (docs/design/plan.md section 5), one mode per scout.sh step.
//   scout info gpu pcie ram     step 2: context/VRAM, GPU read rate, launches, PCIe, doorbell, DDR4
//   scout cpuq                  step 3: ggml's CPU expert dot products (the kernels the copy uses)
//   scout combined <s>          step 5: GPU read loop + CPU expert loop together, one line per second
//   scout hugepages [MiB]       step 6: 4K / THP / 2 MB / 1 GB pages, pinned with cudaHostRegister
// Output lines are "name key=value ...", so the scouting archive parses with awk.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const size_t MiB = 1 << 20;
static const size_t EXPERT = 1153434;  // 1.1 MiB: one expert of the 3.06 bpw file (352.5 MiB / 40 layers / 8)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- CPU topology and threads

static std::vector<int> g_cores, g_sibs;  // first logical CPU of each physical core; the rest

static void topo() {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    for (int c = 0; c < n; c++) {
        char p[128];
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        int first = c;
        if (FILE * f = fopen(p, "r")) { if (fscanf(f, "%d", &first) != 1) first = c; fclose(f); }
        (first == c ? g_cores : g_sibs).push_back(c);
    }
}

static int cpu_for(int i) {  // physical cores first, then their siblings
    std::vector<int> all = g_cores;
    all.insert(all.end(), g_sibs.begin(), g_sibs.end());
    return all[i % all.size()];
}

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
}

// Runs fn(i) on n threads pinned by cpu_for, released together; returns wall seconds.
static double run_threads(int n, const std::function<void(int)> & fn) {
    std::atomic<int> ready{0}; std::atomic<bool> go{false};
    std::vector<std::thread> t;
    for (int i = 0; i < n; i++) t.emplace_back([&, i] { pin(cpu_for(i)); ready++; while (!go) _mm_pause(); fn(i); });
    while (ready < n) _mm_pause();
    double t0 = now(); go = true;
    for (auto & x : t) x.join();
    return now() - t0;
}

// ---------------------------------------------------------------- GPU kernels

__global__ void __launch_bounds__(1024) k_read(const int4 * __restrict__ p, size_t n, int * out, int * smid) {
    if (smid && threadIdx.x == 0) { int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); smid[blockIdx.x] = s; }
    int acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int4 v = __ldg(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x7fffffff) out[0] = acc;  // keeps the loads alive
}

__global__ void k_empty() {}

__global__ void k_bell(volatile int * to_cpu, volatile int * from_cpu, int n) {
    for (int i = 1; i <= n; i++) {
        *to_cpu = i; __threadfence_system();
        while (*from_cpu != i) {}
    }
}

static int g_sms;
static int * g_out;

// Reads `bytes` from p with `blocks` x `threads`; returns GB/s over `reps` launches.
static double gpu_read(const void * p, size_t bytes, int blocks, int threads, int reps, int * smid = nullptr) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    k_read<<<blocks, threads>>>((const int4 *)p, bytes / 16, g_out, smid);
    CK(cudaEventRecord(a));
    for (int r = 0; r < reps; r++) k_read<<<blocks, threads>>>((const int4 *)p, bytes / 16, g_out, nullptr);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return bytes * (double)reps / (ms * 1e6);
}

// ---------------------------------------------------------------- step 2: info

static void mode_info() {
    cudaDeviceProp d; CK(cudaGetDeviceProperties(&d, 0));
    int memclk = 0, bus = 0;
    cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("gpu name=\"%s\" cc=%d.%d sms=%d l2_kib=%d smem_optin_kib=%zu threads_per_sm=%d memclk_mhz=%d bus_bits=%d"
           " peak_gbps=%.1f driver_cuda=%d runtime_cuda=%d\n",
           d.name, d.major, d.minor, d.multiProcessorCount, d.l2CacheSize / 1024, d.sharedMemPerBlockOptin / 1024,
           d.maxThreadsPerMultiProcessor, memclk / 1000, bus, 2.0 * memclk * 1e3 * (bus / 8) / 1e9, drv, rt);
    printf("vram total_mib=%zu free_after_context_mib=%zu used_mib=%zu\n", tot / MiB, fr / MiB, (tot - fr) / MiB);
    printf("cpu cores=%zu siblings=%zu core_cpus=", g_cores.size(), g_sibs.size());
    for (int c : g_cores) printf("%d,", c);
    printf(" sibling_cpus=");
    for (int c : g_sibs) printf("%d,", c);
    printf("\n");
}

// ---------------------------------------------------------------- step 2: GPU

static void mode_gpu() {
    const size_t BUF = 512 * MiB;
    char * buf; CK(cudaMalloc(&buf, BUF)); CK(cudaMemset(buf, 1, BUF));
    int * smid; CK(cudaMalloc(&smid, 1024 * sizeof(int)));

    // Read rate against active SMs: one 1024-thread block fills an SM on Turing and Ampere alike.
    std::vector<int> pts = {1, 2, 4, 6, 9, 12, 18, 24, 30, 36, 48, 60, 68, 84};
    for (int s : pts) {
        if (s > g_sms) break;
        double gbs = gpu_read(buf, BUF, s, 1024, std::max(2, s), smid);  // 20+ ms per point
        std::vector<int> h(s); CK(cudaMemcpy(h.data(), smid, s * sizeof(int), cudaMemcpyDeviceToHost));
        printf("gpu_read_sms sms=%d distinct=%zu gbps=%.1f\n", s, std::set<int>(h.begin(), h.end()).size(), gbs);
    }
    printf("gpu_read_full gbps=%.1f\n", gpu_read(buf, BUF, g_sms * 8, 256, 40));

    // Decode-sized matrices: back-to-back reads of one matrix each, rotating through the buffer so L2 never hits.
    // As separate launches, and as one CUDA graph: the launch gap is what the copy pays per kernel.
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (size_t mb : {1, 2, 4, 8, 17, 64}) {
        size_t sz = mb * MiB; int n = (int)std::min<size_t>(200, BUF / sz);
        auto launch_all = [&] { for (int i = 0; i < n; i++)
            k_read<<<g_sms * 8, 256, 0, st>>>((const int4 *)(buf + (size_t)i * sz % (BUF - sz)), sz / 16, g_out, nullptr); };
        launch_all(); CK(cudaStreamSynchronize(st));
        double t0 = now(); launch_all(); CK(cudaStreamSynchronize(st)); double ts = now() - t0;
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); launch_all(); CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0)); CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
        t0 = now(); CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st)); double tg = now() - t0;
        printf("gpu_read_matrix mib=%zu n=%d stream_gbps=%.1f graph_gbps=%.1f stream_us=%.1f graph_us=%.1f\n",
               mb, n, sz * (double)n / ts / 1e9, sz * (double)n / tg / 1e9, ts / n * 1e6, tg / n * 1e6);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }

    // Launch costs: empty kernels in a stream, empty graph nodes, and a launch + sync round trip.
    const int N = 2000;
    for (int i = 0; i < 100; i++) k_empty<<<1, 32, 0, st>>>();
    CK(cudaStreamSynchronize(st));
    double t0 = now(); for (int i = 0; i < N; i++) k_empty<<<1, 32, 0, st>>>();
    CK(cudaStreamSynchronize(st)); double tl = (now() - t0) / N;
    cudaGraph_t g; cudaGraphExec_t ge; const int NODES = 200;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < NODES; i++) k_empty<<<1, 32, 0, st>>>();
    CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    t0 = now(); for (int i = 0; i < 10; i++) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st)); double tn = (now() - t0) / (10.0 * NODES);
    std::vector<double> rt;
    for (int i = 0; i < 500; i++) { t0 = now(); k_empty<<<1, 32, 0, st>>>(); CK(cudaStreamSynchronize(st)); rt.push_back(now() - t0); }
    std::sort(rt.begin(), rt.end());
    printf("gpu_launch stream_us=%.2f graph_node_us=%.2f launch_sync_p50_us=%.1f launch_sync_p99_us=%.1f\n",
           tl * 1e6, tn * 1e6, rt[rt.size() / 2] * 1e6, rt[rt.size() * 99 / 100] * 1e6);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(st);
    cudaFree(smid); cudaFree(buf);
}

// ---------------------------------------------------------------- step 2: PCIe and doorbell

static void mode_pcie() {
    const size_t BUF = 64 * MiB;
    char *h, *d; CK(cudaHostAlloc(&h, BUF, cudaHostAllocDefault)); CK(cudaMalloc(&d, BUF));
    memset(h, 1, BUF);
    char * pg = (char *)malloc(BUF); memset(pg, 1, BUF);
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    auto rate = [&](void * dst, const void * src, size_t sz, cudaMemcpyKind k) {
        int n = (int)std::max<size_t>(20, 256 * MiB / sz);
        for (int i = 0; i < 3; i++) CK(cudaMemcpyAsync(dst, src, sz, k, st));
        CK(cudaStreamSynchronize(st));
        double t0 = now();
        for (int i = 0; i < n; i++) {
            size_t off = (size_t)i * sz % (BUF - sz + 1) & ~(size_t)4095;
            bool h2d = k == cudaMemcpyHostToDevice;
            CK(cudaMemcpyAsync((char *)dst + (h2d ? off : 0), (const char *)src + (h2d ? 0 : off), sz, k, st));
        }
        CK(cudaStreamSynchronize(st));
        return sz * (double)n / (now() - t0) / 1e9;
    };
    for (size_t sz : {64 << 10, 256 << 10, (int)EXPERT, 4 << 20, 16 << 20, 64 << 20})
        printf("pcie size_kib=%zu h2d_pinned_gbps=%.2f d2h_pinned_gbps=%.2f h2d_pageable_gbps=%.2f\n", sz >> 10,
               rate(d, h, sz, cudaMemcpyHostToDevice), rate(h, d, sz, cudaMemcpyDeviceToHost),
               rate(d, pg, sz, cudaMemcpyHostToDevice));
    std::vector<double> lat;
    for (int i = 0; i < 200; i++) {
        double t0 = now(); CK(cudaMemcpyAsync(d, h, EXPERT, cudaMemcpyHostToDevice, st)); CK(cudaStreamSynchronize(st));
        lat.push_back(now() - t0);
    }
    std::sort(lat.begin(), lat.end());
    printf("pcie_expert_copy_sync p50_us=%.1f p99_us=%.1f\n", lat[100] * 1e6, lat[198] * 1e6);

    // Doorbell: the GPU writes to mapped host memory, a CPU thread on an HT sibling echoes it back.
    int * hm; CK(cudaHostAlloc(&hm, 2 * 64, cudaHostAllocMapped)); memset(hm, 0, 128);
    int * dm; CK(cudaHostGetDevicePointer(&dm, hm, 0));
    volatile int * to_cpu = hm, * from_cpu = hm + 16;
    const int N = 20000;
    std::atomic<bool> stop{false};
    std::thread echo([&] {
        pin(g_sibs.empty() ? cpu_for(1) : g_sibs[0]);
        int seen = 0;
        while (seen < N && !stop) { int v = *to_cpu; if (v != seen) { seen = v; *from_cpu = v; } else _mm_pause(); }
    });
    double t0 = now();
    k_bell<<<1, 1, 0, st>>>(dm, dm + 16, N);
    CK(cudaStreamSynchronize(st));
    double t = now() - t0; stop = true; echo.join();
    printf("doorbell round_trip_us=%.2f n=%d echo_cpu=%d\n", t / N * 1e6, N, g_sibs.empty() ? cpu_for(1) : g_sibs[0]);
    cudaFreeHost(hm); cudaStreamDestroy(st); free(pg); cudaFreeHost(h); cudaFree(d);
}

// ---------------------------------------------------------------- step 2: DDR4

static double ram_read(const char * buf, size_t bytes, int nt, double secs) {
    std::vector<double> got(nt);
    double t = run_threads(nt, [&](int i) {
        size_t sl = bytes / nt & ~(size_t)63; const __m256i * p = (const __m256i *)(buf + sl * i);
        __m256i a0 = _mm256_setzero_si256(), a1 = a0;
        double t0 = now(); size_t done = 0;
        while (now() - t0 < secs) {
            for (size_t j = 0; j < sl / 32; j += 2) {
                a0 = _mm256_add_epi64(a0, _mm256_load_si256(p + j)); a1 = _mm256_add_epi64(a1, _mm256_load_si256(p + j + 1));
            }
            done += sl;
        }
        volatile long long sink = _mm256_extract_epi64(_mm256_add_epi64(a0, a1), 0); (void)sink;
        got[i] = done;
    });
    double sum = 0; for (double g : got) sum += g;
    return sum / t / 1e9;
}

static void mode_ram() {
    const size_t BUF = 2048 * MiB;
    char * buf = (char *)aligned_alloc(4096, BUF); memset(buf, 1, BUF);
    int nc = (int)g_cores.size(), nt = nc + (int)g_sibs.size();
    for (int n : {1, nc, nt}) printf("ram_read threads=%d gbps=%.1f\n", n, ram_read(buf, BUF, n, 0.6));

    // The same 6-thread read while a DMA stream copies pinned memory to the GPU: the shared-DDR4 cap.
    char *h, *d; CK(cudaHostAlloc(&h, 256 * MiB, cudaHostAllocDefault)); CK(cudaMalloc(&d, 256 * MiB));
    memset(h, 1, 256 * MiB);
    std::atomic<bool> stop{false}; std::atomic<size_t> dma{0};
    std::thread t([&] {
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        while (!stop) { CK(cudaMemcpyAsync(d, h, 256 * MiB, cudaMemcpyHostToDevice, st)); CK(cudaStreamSynchronize(st)); dma += 256 * MiB; }
        cudaStreamDestroy(st);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    size_t d0 = dma; double t0 = now();
    double cpu = ram_read(buf, BUF, nc, 1.0);
    double dg = (dma - d0) / (now() - t0) / 1e9;
    stop = true; t.join();
    printf("ram_read_with_dma threads=%d cpu_gbps=%.1f dma_gbps=%.1f sum_gbps=%.1f\n", nc, cpu, dg, cpu + dg);
    cudaFreeHost(h); cudaFree(d); free(buf);
}

// ---------------------------------------------------------------- step 3: ggml CPU expert kernels

struct QBuf { ggml_type t; int K; size_t row, rows; std::vector<char> w, y; };

// A 256 MiB weight set of one type and row length, built from one quantized expert-sized block.
static QBuf qbuf(ggml_type t, int K) {
    QBuf q{t, K, ggml_row_size(t, K), 0, {}, {}};
    const int R = 1024;
    std::vector<float> src((size_t)R * K), imat(K, 1.0f);
    uint32_t s = 12345;
    for (auto & x : src) { s = s * 1664525u + 1013904223u; x = ((s >> 9) / 8388608.0f - 0.5f) * 0.1f; }
    std::vector<char> blk(q.row * R);
    ggml_quantize_chunk(t, src.data(), blk.data(), 0, R, K, imat.data());
    q.rows = 256 * MiB / q.row / R * R;
    q.w.resize(q.rows * q.row);
    for (size_t r = 0; r < q.rows; r += R) memcpy(q.w.data() + r * q.row, blk.data(), blk.size());
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(t);
    q.y.resize(ggml_row_size(tr->vec_dot_type, K));
    ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(src.data(), q.y.data(), K);
    return q;
}

// Thread i dots its share of rows against one activation, pass after pass; bytes counted per finished pass.
static void qdot_worker(const QBuf & q, int i, int nt, std::atomic<bool> & stop, std::atomic<size_t> & bytes) {
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(q.t);
    size_t r0 = q.rows * i / nt, r1 = q.rows * (i + 1) / nt;
    float s, sink = 0;
    while (!stop) {
        for (size_t r = r0; r < r1; r++) { tr->vec_dot(q.K, &s, 0, q.w.data() + r * q.row, 0, q.y.data(), 0, 1); sink += s; }
        bytes += (r1 - r0) * q.row;
    }
    volatile float v = sink; (void)v;
}

static double qdot_rate(const QBuf & q, int nt, double secs) {
    std::atomic<bool> stop{false}; std::atomic<size_t> bytes{0};
    std::thread timer([&] { std::this_thread::sleep_for(std::chrono::duration<double>(secs)); stop = true; });
    double t = run_threads(nt, [&](int i) { qdot_worker(q, i, nt, stop, bytes); });
    timer.join();
    return bytes / t / 1e9;
}

static void mode_cpuq() {
    int nc = (int)g_cores.size();
    for (ggml_type t : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS})
        for (int K : {2048, 512}) {  // gate_up rows have 2048 columns, down rows 512
            double t0 = now(); QBuf q = qbuf(t, K); double tq = now() - t0;
            double r1 = qdot_rate(q, 1, 0.8), rn = qdot_rate(q, nc, 0.8);
            printf("cpu_expert type=%s K=%d gbps_1t=%.2f gbps_%dt=%.2f per_core_at_%dt=%.2f setup_s=%.1f\n",
                   ggml_type_name(t), K, r1, nc, rn, nc, rn / nc, tq);
            fflush(stdout);
        }
}

// ---------------------------------------------------------------- step 5: combined load

static void mode_combined(int secs) {
    int nc = (int)g_cores.size();
    const size_t BUF = 512 * MiB;
    char * buf; CK(cudaMalloc(&buf, BUF)); CK(cudaMemset(buf, 1, BUF));
    QBuf q = qbuf(GGML_TYPE_IQ3_XXS, 2048);
    std::atomic<bool> stop{false}; std::atomic<size_t> gbytes{0}, cbytes{0};
    std::thread gpu([&] {
        cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        while (!stop) {
            for (int i = 0; i < 4; i++) k_read<<<g_sms * 8, 256, 0, st>>>((const int4 *)buf, BUF / 16, g_out, nullptr);
            CK(cudaStreamSynchronize(st)); gbytes += 4 * BUF;
        }
        cudaStreamDestroy(st);
    });
    std::thread cpu([&] { run_threads(nc, [&](int i) { qdot_worker(q, i, nc, stop, cbytes); }); });
    size_t g0 = 0, c0 = 0; double t0 = now();
    for (int s = 1; s <= secs; s++) {
        std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(
            (long)((t0 + s - now()) * 1000)));
        size_t g = gbytes, c = cbytes;
        printf("combined t=%d unix=%.0f gpu_gbps=%.1f cpu_iq3xxs_gbps=%.2f\n", s,
               (double)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(),
               (g - g0) / 1e9, (c - c0) / 1e9);
        fflush(stdout); g0 = g; c0 = c;
    }
    stop = true; gpu.join(); cpu.join(); cudaFree(buf);
}

// ---------------------------------------------------------------- step 6: page sizes

static void mode_hugepages(size_t mib) {
    size_t sz = mib * MiB;
    int nc = (int)g_cores.size();
    char * d; CK(cudaMalloc(&d, 64 * MiB));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    struct K { const char * name; int flags; int adv; } kinds[] = {
        {"4k", 0, MADV_NOHUGEPAGE}, {"thp", 0, MADV_HUGEPAGE},
        {"2m", MAP_HUGETLB | MAP_HUGE_2MB, -1}, {"1g", MAP_HUGETLB | MAP_HUGE_1GB, -1}};
    for (auto & k : kinds) {
        char * p = (char *)mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | k.flags, -1, 0);
        if (p == MAP_FAILED) { printf("pages kind=%s unavailable errno=%d (%s)\n", k.name, errno, strerror(errno)); continue; }
        if (k.adv >= 0) madvise(p, sz, k.adv);
        double t0 = now(); memset(p, 1, sz); double tf = now() - t0;
        t0 = now(); cudaError_t e = cudaHostRegister(p, sz, cudaHostRegisterDefault); double tr = now() - t0;
        if (e != cudaSuccess) { printf("pages kind=%s register_failed=%s\n", k.name, cudaGetErrorString(e)); munmap(p, sz); continue; }
        // Expert-sized copies and reads from random offsets: the arena access pattern, where TLB reach matters.
        const int N = 1500; std::vector<size_t> off(N); uint32_t s = 7;
        for (auto & o : off) { s = s * 1664525u + 1013904223u; o = ((size_t)s * 4096) % (sz - EXPERT) & ~(size_t)4095; }
        t0 = now();
        for (int i = 0; i < N; i++) CK(cudaMemcpyAsync(d + (size_t)i % 48 * EXPERT, p + off[i], EXPERT, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st)); double th = now() - t0;
        std::atomic<size_t> rd{0};
        double tc = run_threads(nc, [&](int i) {
            __m256i a = _mm256_setzero_si256();
            for (int j = i; j < N; j += nc) {
                const __m256i * q = (const __m256i *)(p + off[j]);
                for (size_t x = 0; x < EXPERT / 32; x++) a = _mm256_add_epi64(a, _mm256_loadu_si256(q + x));
                rd += EXPERT;
            }
            volatile long long v = _mm256_extract_epi64(a, 0); (void)v;
        });
        printf("pages kind=%s mib=%zu fault_in_s=%.2f register_s=%.3f h2d_random_expert_gbps=%.2f cpu_random_expert_gbps=%.1f\n",
               k.name, mib, tf, tr, EXPERT * (double)N / th / 1e9, rd / tc / 1e9);
        fflush(stdout);
        cudaHostUnregister(p); munmap(p, sz);
    }
    cudaStreamDestroy(st); cudaFree(d);
}

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: scout info|gpu|pcie|ram|cpuq|combined <s>|hugepages [MiB]|hold <free MiB> ...\n"); return 2; }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    topo();
    ggml_cpu_init();
    CK(cudaFree(0));  // create the context first, so "info" sees what it costs
    CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaMalloc(&g_out, 64));
    for (int i = 1; i < argc; i++) {
        std::string m = argv[i];
        double t0 = now();
        if (m == "info") mode_info();
        else if (m == "gpu") mode_gpu();
        else if (m == "pcie") mode_pcie();
        else if (m == "ram") mode_ram();
        else if (m == "cpuq") mode_cpuq();
        else if (m == "combined") mode_combined(i + 1 < argc ? atoi(argv[++i]) : 60);
        else if (m == "hugepages") mode_hugepages(i + 1 < argc && isdigit(argv[i + 1][0]) ? atoi(argv[++i]) : 2048);
        else if (m == "hold") {   // proxy: hold VRAM until this much stays free (the laptop's budget), until killed
            const size_t want_free = (size_t) (i + 1 < argc ? atoll(argv[++i]) : 7800) * MiB;
            size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
            void* p = nullptr;
            if (fr > want_free) CK(cudaMalloc(&p, fr - want_free));
            CK(cudaMemGetInfo(&fr, &tot));
            printf("hold free_mib=%zu\n", fr / MiB);
            fflush(stdout);
            pause();
        }
        else { fprintf(stderr, "unknown mode %s\n", m.c_str()); return 2; }
        printf("mode_done mode=%s seconds=%.1f\n", m.c_str(), now() - t0);
    }
    return 0;
}
