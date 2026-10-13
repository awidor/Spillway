// CPU only; no CUDA headers/runtime, model files or device queries.
// Also built as the pcie_auto_test CMake target with STRATA_BUILD_TESTS=ON (checks stay on in Release).
#include "strata/program/pcie_auto.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using strata::program::PcieAuto;
static int checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// n decode windows of ms_step pool milliseconds and jobs_step missed experts each -> the last new share (-1: none)
static double advance(PcieAuto& a, int n, double ms_step, int jobs_step, double cur) {
    double ms = a.mark_ms, answer = -1.0, tc = 0.0, rate = 0.0;
    int64_t jobs = a.mark_jobs;
    for (int i = 0; i < n; ++i) {
        ms += ms_step;
        jobs += jobs_step;
        const double f = a.window(ms, jobs, cur, &tc, &rate);
        if (f >= 0.0) answer = f;
    }
    return answer;
}

static PcieAuto on_12gbps() {
    PcieAuto a;
    a.on = true;
    a.link_gbps = 12.0;
    a.blob_bytes = 1e6;
    return a;
}

int main() {
    PcieAuto a = on_12gbps();
    a.on = false;
    check(advance(a, 60, 10, 10, 0.33) < 0, "off: the share is left alone");

    a.on = true;
    a.mark(900, 900);   // the prompt's pool work, before the first decode window
    check(advance(a, 47, 10, 10, 0.33) < 0, "no change before 48 windows");
    const double f = advance(a, 1, 10, 10, 0.33);
    check(f >= 0.10 && f <= 0.60, "the new share stays in [0.10, 0.60]");
    check(std::fabs(f * 20 - std::round(f * 20)) < 1e-9, "the new share is a multiple of 0.05");
    advance(a, 96, 10, 10, 0.33);
    advance(a, 96, 10, 10, 0.33);
    check(a.updates == 3 && !a.on, "three updates, then it stops");

    PcieAuto hits = on_12gbps();
    check(advance(hits, 2000, 0, 0, 0.33) < 0 && !hits.on, "no missed experts: no reading, off after 2000 windows");

    PcieAuto no_time = on_12gbps();
    check(advance(no_time, 48, 0, 10, 0.33) < 0, "zero pool time: no division by zero, no change");

    std::printf("pcie_auto_test: %d checks passed\n", checks);
}
