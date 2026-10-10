"""Speed model: Qwen3.6-35B-A3B (byteshape GGUF) on the Acer Predator Triton 500 PT515-51.

There is little hardware access, so this projects decode and prefill speed from the bytes each token moves.
Four engines are modelled:
  llama  - llama.cpp with --n-cpu-moe only. Whole layers' experts sit on the CPU, the GPU waits for them.
  cache  - llama.cpp with --moe-cache-mib (PR #29887, merged 2026-10-07). An LRU expert cache in VRAM, and the
           misses are uploaded over PCIe and computed on the GPU.
  strata - the copied Strata port (#1820). The same cache, and each layer's misses are shared between the CPU
           (computed in place in RAM) and PCIe uploads, so both finish together; charged copy_eff for its gap.
  tuned  - the copy after PLAN.md's phases 3-5: fewer launches per layer and the port gap recovered. Same files and
           same ggml CPU kernels: nothing is re-quantized.
selftest() checks six published measurements: Strata's paper (with and without MTP), a llama.cpp RTX 2070 run,
llama.cpp's expert cache on an RTX 3070 Laptop, the #1402 port and byteshape's GPU-only RTX 5060 Ti numbers.

    python perfmodel/speed_model.py              # tables for all engines, every file, MTP on/off
    python perfmodel/speed_model.py --ctx 8192
    python perfmodel/speed_model.py --selftest   # calibration checks only

Every input is a named constant. Replace guesses with the S1 scouting results and phase 2 (PLAN.md).
"""
import argparse

MiB = 2 ** 20
GB = 1e9

# --- model geometry: Qwen/Qwen3.6-35B-A3B config.json, confirmed by the GGUF headers ---------------------------
ARCH = dict(n_layers=40, n_attn_layers=10, n_expert=256, n_used=8, n_kv_head=2, head_dim=256,
            active_params=2.95e9, gdn_state_mib=63)

# byteshape files, per-category sizes parsed from each GGUF header (MiB).
#   experts = routed experts of the 40 main layers; dense = everything else read per token (incl. the output
#   head, excl. token_embd); mtp = the MTP layer blk.40 (528 MiB of it is Q5_K experts); tok = routed-expert
#   MiB one token reads (8 experts x 40 layers). draft = MiB one MTP draft reads: the MTP layer's dense part (39) +
#   8 Q5_K experts (17) + the output head, full (186) or cut to a ~40K-token draft vocabulary (30).
QUANTS = {
    "2.25bpw": dict(experts=7920, dense=870, mtp=567, tok=247.5),
    "3.06bpw": dict(experts=11280, dense=936, mtp=567, tok=352.5),
    "3.53bpw": dict(experts=13032, dense=1148, mtp=567, tok=407.2),
    "3.97bpw": dict(experts=15028, dense=1031, mtp=567, tok=469.6),
    "4.19bpw": dict(experts=15870, dense=1110, mtp=567, tok=495.9),
}

# --- the laptop (hardware profile PDF, 27.09.2026) ---------------------------------------------------------------
LAPTOP = dict(
    vram_mib=8192, vram_os_mib=300,   # headless, dedicated: only our CUDA context (persistence mode, lazy loading)
    gpu_bw=384 * GB,                  # RTX 2070 Max-Q: 256-bit, 12 Gbps GDDR6 (notebookcheck); OEM clock unverified
    ram_mib=16384, ram_os_mib=1200,   # headless Linux, nothing else running, thin server; check with `free -m`
    ram_bw=42.7 * GB,                 # DDR4-2667, 2 channels, theoretical
    pcie_bw=12.0 * GB,                # PCIe 3.0 x16 pinned DMA, typical measured (15.75 theoretical)
    cpu_cores=6,
)

# --- rate scenarios (lo = pessimistic, hi = optimistic) ----------------------------------------------------------
# gpu_eff      share of nominal VRAM bandwidth that decode GEMVs reach (Strata 5070 calibrates to ~0.5;
#              Strata's RTX 2070 GEMV bench reached 0.62-0.83, on large shapes only)
# layer_us     fixed per-layer cost: launches + CPU/GPU handshake (strata: one doorbell per layer, CUDA graph)
# gpu_us       llama.cpp's per-layer cost with every weight on the GPU (byteshape's 5060 Ti runs: ~80 us)
# split_us     extra per layer when experts run on the CPU (a graph split and sync at every layer)
# sync_us      extra per layer for #29887's cache (the expert ids are read back before misses upload)
# copy_eff     the copied Strata port's throughput on qwen35moe relative to these equations: fitted to #1402's
#              measurement (selftest 4). The tuned engine assumes phases 3-4 recover it.
# cpu_core_gbs expert-weight GB/s one i7-9750H core sustains on i-quants (Strata measured ~5 GB/s/core on
#              Zen 4 at ~5 GHz; the 9750H holds ~3.0-3.4 GHz all-core at 45 W, and ggml's AVX2 i-quant dots
#              are compute-bound)
# hit_exp      static hit curve h = f^hit_exp. Strata's Figure 3 (Flash-Next) has h = 0.50 at f = 0.18 (exp 0.41) and
#              ~0.93 at f = 0.53, so even 0.35 is conservative there; this model's curve is unmeasured (phase 2)
# hit_boost    how much of the remaining miss rate the adaptive swap recovers (Strata: 0.50 -> 0.72 = 0.44)
# p_acc        geometric acceptance per draft position. #1402 reports 2.4-3.1 tokens per 4-token window on this
#              model (p ~ 0.62-0.80); its 0.84-0.96 'accepted' counts only drafts the min-p gate let through
# ggml_cpu     ggml's CPU MoE path relative to Strata's in-place expert kernels (per-op threading, activation
#              re-quantization, i-quants on AVX2; K-quants are not penalized); fitted so the RTX 3070 Laptop baseline lands near 35 tok/s
# pcie_eff     share of the PCIe rate small per-expert uploads reach (pageable mmap memory is slower)
# tuned_us     per-layer fixed cost of the tuned engine (<= 6 fused kernels per layer in one CUDA graph)
# pp_tflops    effective GPU throughput on prompt chunks (dequant + MMQ + launches), TFLOP/s
SCEN = {
    "lo":  dict(gpu_eff=0.40, layer_us=140, gpu_us=100, split_us=150, sync_us=60, copy_eff=0.83, cpu_core_gbs=2.2, ram_eff=0.62, hit_boost=0.00, hit_exp=0.55,
                p_acc=0.65, pcie_eff=0.55, ggml_cpu=0.40, tuned_us=90, pp_tflops=3.0),
    "mid": dict(gpu_eff=0.50, layer_us=100, gpu_us=80, split_us=100, sync_us=40, copy_eff=0.83, cpu_core_gbs=2.8, ram_eff=0.70, hit_boost=0.25, hit_exp=0.45,
                p_acc=0.75, pcie_eff=0.70, ggml_cpu=0.50, tuned_us=60, pp_tflops=5.0),
    "hi":  dict(gpu_eff=0.55, layer_us=80, gpu_us=60, split_us=60, sync_us=25, copy_eff=0.83, cpu_core_gbs=3.4, ram_eff=0.78, hit_boost=0.44, hit_exp=0.35,
                p_acc=0.80, pcie_eff=0.85, ggml_cpu=0.65, tuned_us=60, pp_tflops=8.0),
}


def static_hit(f, e=0.45):
    """Hit rate of a profile-filled cache holding share f of all experts.

    shortcut: a power law below Strata's Figure 3 (512 experts, top-10). Replace it with the curve measured from
    qwen35moe routing traces (PLAN.md, phase 2)."""
    return min(1.0, f ** e) if f > 0 else 0.0


def unique_frac(k, a=ARCH):
    """Unique experts per layer touched by k tokens, relative to k * n_used (independent routing)."""
    n, u = a["n_expert"], a["n_used"]
    return n * (1 - (1 - u / n) ** k) / (k * u)


def fit(q, hw=LAPTOP, ctx=32_768, mtp=True, a=ARCH):
    kv_mib = (a["n_attn_layers"] + mtp) * 2 * a["n_kv_head"] * a["head_dim"] * 1.0625 * ctx / MiB  # q8_0 K and V
    work_mib = 600                                                       # activations, prompt ring, graphs
    cache = (hw["vram_mib"] - hw["vram_os_mib"] - q["dense"] - kv_mib - a["gdn_state_mib"] - work_mib
             - (q["mtp"] if mtp else 0))
    cache = max(cache, 0)
    # RAM holds what VRAM does not (Strata's low-RAM mode / llama.cpp mmap), plus token_embd and buffers
    ram_need = q["experts"] - cache + 186 + 400
    return dict(cache=cache, f=cache / q["experts"], ram_need=ram_need,
                ram_ok=ram_need <= hw["ram_mib"] - hw["ram_os_mib"])


def decode_tps(q, s, engine, hw=LAPTOP, ctx=32_768, k=1, a=ARCH):
    """Tokens/s with verify windows of k tokens (k=1: plain decoding, no MTP layer loaded)."""
    m = fit(q, hw, ctx, mtp=k > 1, a=a)
    if m["f"] >= 1:
        hit = 1.0                                        # every expert fits in VRAM
    elif engine == "llama":
        hit = m["f"]                                     # whole layers on the GPU: the hit rate is the layer share
    else:
        h0 = static_hit(m["f"], s["hit_exp"])
        hit = min(0.97, h0 + s["hit_boost"] * (1 - h0))
    win = q["tok"] * MiB * k * unique_frac(k, a)        # routed-expert bytes one window touches
    gpu_bw = hw["gpu_bw"] * s["gpu_eff"]
    t_dense = q["dense"] * MiB / gpu_bw                 # read once per window
    t_gpu_exp = hit * win / gpu_bw
    # DeltaNet state (f32, read + written) and the q8 KV cache: read once per window
    t_attn = (a["gdn_state_mib"] * 2 * MiB + a["n_attn_layers"] * 2 * a["n_kv_head"] * a["head_dim"] * 1.0625 * ctx) / gpu_bw
    cpu_rate = min(hw["cpu_cores"] * s["cpu_core_gbs"] * GB, hw["ram_bw"] * s["ram_eff"])
    pcie = hw["pcie_bw"] * s["pcie_eff"]
    miss = hit < 1
    us = {"llama": s["gpu_us"] + s["split_us"] * miss, "cache": s["gpu_us"] + s["sync_us"] * miss,
          "strata": s["layer_us"], "tuned": s["tuned_us"]}[engine] * 1e-6
    if engine == "llama":
        t_cpu = (1 - hit) * win / cpu_rate / (s["ggml_cpu"] if q.get("iq", True) else 1.0)
        t = t_dense + t_gpu_exp + t_cpu
    elif engine == "cache":
        t = t_dense + win / gpu_bw + (1 - hit) * win / pcie
    else:
        ram = hw["ram_bw"] * s["ram_eff"]
        rate = min(cpu_rate + pcie, ram)                 # CPU reads and DMA reads share the same DDR4
        t_cpu = (1 - hit) * win / rate
        t_gpu_exp += (1 - hit) * win * (pcie / (cpu_rate + pcie)) / gpu_bw
        # Strata Table 5: the CPU waits for the mixer, then the experts run on the GPU and the CPU at once
        t = t_dense + max(t_gpu_exp, t_cpu)
    t += t_attn + a["n_layers"] * us
    if k > 1:                                            # k-1 sequential MTP drafts; Strata drafts on a ~40K vocab
        head = q.get("head", 186) * (40 / 248 if engine in ("strata", "tuned") else 1.0)
        t += (k - 1) * ((q.get("draft", 56) + head) * MiB / gpu_bw + 2 * us)
    if engine == "strata" and a is ARCH:
        t /= s["copy_eff"]
    tokens = sum(s["p_acc"] ** i for i in range(k))
    return tokens / t, hit, m


def prefill_tps(q, s, engine, hw=LAPTOP, ctx=32_768, chunk=2048, a=ARCH):
    """Prompt tokens/s: missed experts stream over PCIe while the GPU computes the chunk (both engines do this;
    llama.cpp copies CPU-held experts to the GPU for large batches)."""
    m = fit(q, hw, ctx, mtp=False, a=a)
    t_stream = (q["experts"] - m["cache"]) * MiB / hw["pcie_bw"]
    t_compute = chunk * 2 * a["active_params"] / (s["pp_tflops"] * 1e12)
    # the cache PR only acts on batches <= 32 tokens, so prompts take the regular offload path in both llama modes
    t = max(t_stream, t_compute) if engine in ("strata", "tuned") else t_stream + t_compute
    return chunk / t


def selftest():
    # 1) Strata's machine: Q2_0, 4K, measured 94.6 tok/s at 3.23 tokens per window, hit 0.72 (paper Tables 3-5)
    strata = dict(n_layers=48, n_attn_layers=12, n_expert=512, n_used=10, n_kv_head=2, head_dim=256,
                  active_params=6e9, gdn_state_mib=150)
    pc = dict(vram_mib=12288, vram_os_mib=800, gpu_bw=672 * GB, ram_mib=65536, ram_os_mib=6000, ram_bw=83 * GB,
              pcie_bw=26 * GB, cpu_cores=6)
    # drafting measured at 2.0 ms per round of 3 drafts (Table 5): ~200 MiB per draft at the calibrated bandwidth
    q2 = dict(experts=34000 * 0.9537, dense=3500 * 0.9537, mtp=800, tok=0.66 * 1000 * 0.9537, draft=170, head=200)
    s = dict(SCEN["mid"], cpu_core_gbs=7.0, ram_eff=0.5, hit_boost=0.44, p_acc=0.82)
    tps, hit, _ = decode_tps(q2, s, "strata", pc, ctx=4096, k=4, a=strata)
    plain = decode_tps(q2, s, "strata", pc, ctx=4096, k=1, a=strata)[0]
    print(f"selftest strata: RTX 5070 Q2_0 4K projected {tps:.0f} tok/s (measured 94.6), hit {hit:.2f} (0.72), "
          f"without MTP {plain:.0f} (measured 47-57)")
    assert 75 < tps < 115 and 0.6 < hit < 0.85 and 40 < plain < 65, (tps, hit, plain)
    # 2) community llama.cpp: RTX 2070 8 GB (448 GB/s) + 32 GB DDR4, "Q6 XL" (~6.6 bpw) ~18 tok/s, flags unknown
    d2070 = dict(LAPTOP, gpu_bw=448 * GB, ram_mib=32768, ram_bw=51 * GB)
    q6 = dict(experts=25350, dense=1700, mtp=900, tok=792, iq=False)
    lo, mid, hi = (decode_tps(q6, SCEN[n], "llama", d2070, ctx=8192)[0] for n in ("lo", "mid", "hi"))
    print(f"selftest llama:  RTX 2070 8 GB Q6 projected {lo:.0f}/{mid:.0f}/{hi:.0f} tok/s lo/mid/hi (reported ~18)")
    assert lo <= 18 * 1.25 and hi >= 18 * 0.75, (lo, hi)
    # 3) llama.cpp PR #30059 (open; a cache like #29887): RTX 3070 Laptop 8 GB (448 GB/s) + 32 GB, this model,
    #    IQ2_M 35.0 -> 82.2 tok/s with the cache. IQ2_M approximated by the 2.25bpw file.
    d3070 = dict(LAPTOP, gpu_bw=448 * GB, ram_mib=32768, ram_bw=51 * GB, pcie_bw=12.0 * GB)
    q = QUANTS["2.25bpw"]
    base = [decode_tps(q, SCEN[n], "llama", d3070, ctx=8192)[0] for n in ("lo", "hi")]
    cache = [decode_tps(q, SCEN[n], "cache", d3070, ctx=8192)[0] for n in ("lo", "hi")]
    print(f"selftest cache:  RTX 3070 Laptop 2.25bpw projected {base[0]:.0f}-{base[1]:.0f} -> "
          f"{cache[0]:.0f}-{cache[1]:.0f} tok/s (reported 35.0 -> 82.2 at IQ2_M)")
    # 4) Strata PR #1402 (qwen35moe port): RTX 4070 Ti (504 GB/s) with 4 GiB held -> sized like an 8 GB card,
    #    Ryzen 9 5950X, UD-IQ3_S (12.0 GiB experts), MTP --spec 4, drafts accepted 0.84-0.96:
    #    short chat 89 tok/s, writes after an 8K prompt 70 tok/s (docs/QWEN36.md on that branch)
    #    That run kept only 2,003 of 10,240 experts (2.4 GiB) in VRAM: Strata's buffers, desktop and BF16-packed dense
    #    weights took ~3 GiB more than fit() assumes. vram_os_mib reproduces the measured cache size.
    d4070 = dict(LAPTOP, gpu_bw=504 * GB, vram_os_mib=3080, ram_mib=63488, ram_bw=51 * GB, pcie_bw=24 * GB,
                 cpu_cores=16)
    uds = dict(experts=12288, dense=1300, mtp=600, tok=384)
    pr = [decode_tps(uds, dict(SCEN[n], p_acc=0.80, cpu_core_gbs=4.0), "strata", d4070, ctx=8192, k=4)
          for n in ("lo", "hi")]
    print(f"selftest #1402:  RTX 4070 Ti as 8 GB, UD-IQ3_S projected {pr[0][0]:.0f}-{pr[1][0]:.0f} tok/s, "
          f"f {pr[0][2]['f']:.2f} (measured 70-89, f 0.20)")
    pr = [x[0] for x in pr]
    # 5) byteshape blog: this model, 3.00 bpw (NTP release) all in VRAM on an RTX 5060 Ti 16 GB (448 GB/s), llama.cpp:
    #    120.7 tok/s; the 3.06 bpw MTP file: 169.5 tok/s. An upper bound: a hybrid laptop cannot beat its GPU-only self.
    d5060 = dict(LAPTOP, vram_mib=16384, gpu_bw=448 * GB)
    q = QUANTS["3.06bpw"]
    ntp = [decode_tps(q, SCEN[n], "cache", d5060, ctx=4096)[0] for n in ("lo", "hi")]
    mtp = [max(decode_tps(q, SCEN[n], "cache", d5060, ctx=4096, k=k)[0] for k in (2, 3, 4)) for n in ("lo", "hi")]
    print(f"selftest 5060Ti: GPU-only llama.cpp projected {ntp[0]:.0f}-{ntp[1]:.0f} (measured 120.7), "
          f"MTP {mtp[0]:.0f}-{mtp[1]:.0f} (measured 169.5)")
    assert ntp[0] < 120.7 * 1.2 and ntp[1] > 120.7 * 0.8 and mtp[0] < 169.5 * 1.2 and mtp[1] > 169.5 * 0.8, (ntp, mtp)
    assert pr[0] < 89 * 1.2 and pr[1] > 70 * 0.8, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ctx", type=int, default=32_768)
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return
    ctx = args.ctx
    print(f"\nlaptop, context {ctx}; decode tok/s lo/mid/hi; MTP column = best draft length k (verify window)")
    for engine in ("llama", "cache", "strata", "tuned"):
        print(f"\n[{engine}]\n{'file':8} {'cache':>6} {'f':>5} {'hit':>5} {'RAM':>6}  {'no MTP':>14}  "
              f"{'MTP (best k)':>23}  {'prefill':>13}")
        for name, q in QUANTS.items():
            m = fit(q, ctx=ctx, mtp=False)
            nm = [decode_tps(q, SCEN[n], engine, ctx=ctx)[0] for n in ("lo", "mid", "hi")]
            mt = [max((decode_tps(q, SCEN[n], engine, ctx=ctx, k=k)[0], k) for k in (2, 3, 4))
                  for n in ("lo", "mid", "hi")]
            hit = decode_tps(q, SCEN["mid"], engine, ctx=ctx)[1]
            pp = [prefill_tps(q, SCEN[n], engine, ctx=ctx) for n in ("lo", "hi")]
            flag = "" if m["ram_ok"] else " RAM!"
            print(f"{name:8} {m['cache'] / 1024:5.1f}G {m['f']:5.2f} {hit:5.2f} {m['ram_need'] / 1024:5.1f}G  "
                  f"{'/'.join(f'{x:.0f}' for x in nm):>14}  {'/'.join(f'{x:.0f}({k})' for x, k in mt):>23}  "
                  f"{pp[0]:5.0f}-{pp[1]:<5.0f}{flag}")


if __name__ == "__main__":
    main()
