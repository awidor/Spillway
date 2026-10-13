#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace strata::program {
struct PcieAuto {
    static constexpr double kPoolFactor = 1.85;
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double mark_ms = 0.0;
    int64_t mark_jobs = 0;
    double acc_ms = 0.0;
    int64_t acc_jobs = 0;
    int acc_win = 0, updates = 0;
    void mark(double ms_run, int64_t jobs) { mark_ms = ms_run; mark_jobs = jobs; }
    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        mark(ms_run, jobs);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_win = 0;
        if (++updates >= 3) on = false;
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;
        const double f = std::min(0.60, std::max(0.10, std::round(link_gbps / (link_gbps + kPoolFactor * r) * 20.0) / 20.0));
        *tc_us = tc;
        *rate = r;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};
}
