# Spillway: a copy of Strata, tuned for one model on one box

**Model:** Qwen3.6-35B-A3B, from [byteshape/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF).

**Machine:** Acer Predator Triton 500 PT515-51. RTX 2070 Max-Q 8 GB, i7-9750H, 16 GB of DDR4-2667, PCIe 3.0 x16,
headless Linux, dedicated to this job. The hardware profile is dated 27.09.2026.

**Approach: copy Strata, cut it down, tune it for this box.**

- **The starting point:** a copy of [Strata](https://github.com/Niko1221/Strata) at the head of
  [#1820](https://github.com/Niko1221/Strata/pull/1820) (`650cf81`).
  - That head already contains upstream `main` @ `fb58e0d` (checked) and [#1402](https://github.com/Niko1221/Strata/pull/1402),
    which teaches the engine this exact architecture (`qwen35moe`).
  - What comes with it: the expert cache, CPU expert pool, doorbell, DMA miss share, adaptive swaps, MTP verify
    windows, draft vocab, CPU prompt share, KV streaming, the arena on 2 MB huge pages, the server and the parity
    tests.
- **What #1402 measured:** 70-89 tok/s with MTP, on an RTX 4070 Ti limited to 8 GB. That machine is far stronger
  than the box (504 GB/s, a 16-core Ryzen 9 5950X, PCIe 4.0). It is a reference point, not a box number.
- **Our work:**
  - the fixes byteshape's files need;
  - the changes that only pay on *this* GPU, CPU, memory and link.

  Every change is measured on the proxy and checked with a parity run. The winner becomes the only code path.
- **A copy, not a fork:** nothing goes upstream, so the copy never has to stay mergeable. We delete what the
  box doesn't run.
- **Access:** we have little time on the box, so every speed below is a projection from
  `perfmodel/speed_model.py`. The model is checked against six published measurements (section 4). Box time
  goes first to a short, scripted scouting session (section 5), which replaces as many guesses as possible with
  measurements.

---

## 0. Status (10 Oct)

All dev-PC phases are done; what is left needs the box. Measured numbers: [RESULTS.md](RESULTS.md).

- **Phase 0:** build container (`kit/Dockerfile`), S1 kit (`out/s1kit/`, kept as the llama.cpp baseline).
- **S1:** skipped (supervised access: build first, measure on the box).
- **Phase 1:** `engine/` = Strata `650cf81`; byteshape's fused `ffn_gate_up_exps` read in place (packer + engine),
  IQ2_XXS down kernels, a missing include fixed; SYCL, translations, Windows launchers and Docker files stripped.
  Greedy parity with llama.cpp: identical on the code prompt, near-tie splits elsewhere.
- **Phases 2-4:** proxy at the box's budget (VRAM held to 7.8 GB, sm_75 paths, 6 cores): 142.8 tok/s with MTP.
  Expert layout mode 8 made the default (VRAM-hit experts 5.1 -> 1.7 ms/window, decode 73 -> 114 tok/s without
  MTP, output identical). #1821 and #1825 merged (parity pass); #1437 and #1581 not taken (RESULTS.md says why).
- **Phase 5:** BF16 MMVQ so byteshape's own MTP layer runs (no re-quantization): 148-167 tok/s on the proxy, k=3.
- **Phase 6 (server + S2 kit):** `out/s2kit/` - `bash s2kit/s2.sh` installs to /opt/strata, packs, tunes
  (draft length, VRAM reserve, pool workers, expert layout), starts `strata.service` (OpenAI API, port 8095) and
  checks it; `--undo` removes it. Dry run on the proxy passed. Kernel command-line settings (2b) are not applied:
  unmeasured, and each needs a reboot.
- **S2 on the box:** not run yet. Copy `out/s2kit/` (and the model, or let it download), run `bash s2kit/s2.sh`;
  for the llama.cpp number on the same box, `bash s1kit/s1.sh` (step 4b).

## 1. Summary

Projected decode speed, in tok/s at a 32K context with the 3.06 bpw file. Each cell is lo / mid / hi; plan with mid.

| Engine | No MTP | Best MTP |
|---|---|---|
| llama.cpp, `--n-cpu-moe` | 19 / 27 / 39 | 17 / 27 / 41 |
| llama.cpp, `--moe-cache-mib` (#29887, merged 7 Oct) | 29 / 44 / 62 | 28 / 48 / 77 |
| The copy as-is, at the 20% expert cache #1402 got on its "8 GB" card | 34 / 50 / 66 | 38 / 67 / 110 |
| **Spillway: the copy plus the box changes** | **53 / 78 / 91** | **63 / 115 / 173*** |

\* **Upper bound.** Byteshape measured this model fully in VRAM on an RTX 5060 Ti at 120.7 tok/s (169.5 with
MTP). Scaled to the box's 384 GB/s, that is about 103 / 145. A hybrid engine can't beat its own all-in-VRAM speed,
so the "hi" MTP value (173) is not credible. Mid sits at 79% of the bound.

**Where the gain comes from (model, mid, with MTP, starting from the copy as-is at 67 tok/s):**

| Step | tok/s after | Gain | Phase |
|---|---|---|---|
| VRAM accounting, 20% -> 47% of experts cached | 89 | +33% | 3 |
| Turing kernels: recover the port's 17% gap to Strata's own model path | 107 | +20% | 4 |
| Fewer launches per layer | 115 | +7% | 4 |
| The dedicated-box OS setup | not modelled | | S2 |

Two caveats:
- **The first step may be partly free.** #1402's 20% was measured on a desktop card that also drove a display. On a
  headless box the copy may already cache more. Session S1 plus phase 2 measure it.
- **Prompt reading:** about 500-1,350 tok/s.

**The box is dedicated and headless:** no display, no desktop, no other service. All 8 GB of VRAM except the CUDA
context, about 14.8 GB of RAM, all 6 cores and the whole power budget belong to the engine.

**Model file: the 3.06 bpw file.** It is 13.6 GB, with 11 GiB of experts.
- **RAM:** all experts are pinned, so the adaptive cache swaps from RAM by DMA and never touches the SSD.
- **Speed fallback:** 2.25 bpw.
- **Quality mode:** 3.97 bpw fits in low-RAM mode, about 58 tok/s without MTP and 70 with (mid). This assumes
  low-RAM mode keeps the VRAM set fixed, which phase 2 checks.
- **If quality ever matters more than speed:** two 16 GB SO-DIMMs (the firmware reports a 32 GiB maximum) would let
  3.97 bpw pin fully.

**Box time:** two short supervised sessions: S1 (scouting) of at most 10 minutes, and S2
(deployment and on-device tuning) of at most 20 minutes. Everything else happens on the dev PC.

## 2. The hardware, and what each fact decides

Each row is a target for the engine. Where the copy already meets it, nothing changes. Where it doesn't, the phase
in brackets changes it. *S1* marks a fact that scouting session 1 measures.

| Fact | Engine decision |
|---|---|
| **TU106 Max-Q, sm_75, 36 SMs, 384 GB/s** (256-bit, 12 Gbps per notebookcheck; the box's real memory clock is *S1*) | Decode is bandwidth-bound: every kernel is judged in achieved GB/s. Grids are sized in multiples of 36 SMs, so there are no tail waves (4). |
| **64 KB shared memory per block (opt-in), 4 MB L2** | Prompt-path int8 tiles are sized for 64 KB, not for Ampere's 99 KB (4). |
| **No BF16 tensor cores; FP16 `mma.m16n8k8`, int8 `mma.m8n8k16` and `dp4a` are present** | Every BF16 tensor is converted offline to FP16, or kept in FP32 where FP16's range is too small. In the 3.06 bpw file those are only in the MTP layer; the 3.53 bpw file also has BF16 attention K/V and shared experts. Decode GEMVs use no tensor cores, because Strata measured tensor-core GEMV on an RTX 2070 1.3-2.2x *slower* (3-4). |
| **hidden 2048: every matrix except the 186-258 MiB output head is small (2-17 MB)** | Launch overhead rivals the reads. Target: at most 6 kernels per layer in one CUDA graph per window size. Count the copy's launches first (4). The doorbell round trip and the graph launch cost per node are *S1*. |
| **DeltaNet state = 128x128 f32 per V head = 64 KB, exactly the sm_75 shared-memory limit** | Target: keep a head's state in registers across a verify window (2 blocks x 256 threads, 32 floats each), so it is read and written once per window. Check what the copied kernel does (4). |
| **8 GB VRAM, headless** | The budget is the whole card minus the CUDA context (its size is *S1*). Prompt buffers are borrowed from the expert cache. Target: at least 5.5 GiB of expert cache at 32K context (3). |
| **PCIe 3.0 x16, about 12-13 GB/s DMA** (a 2080 Ti on PCIe 3.0 x16 measured 13.1 in Strata's own probe; this link's real rate at 1.2 MB expert-sized copies is *S1*) | Decode: a measured share of each layer's misses goes over DMA, so the CPU and the copy engine finish together. Prompts: uncached experts stream (5.2 GiB per chunk, about 0.45 s), so chunks are compute-bound, not link-bound. |
| **i7-9750H: Coffee Lake (Skylake core), 6C/12T, AVX2+FMA, no AVX-512, 45 W** | ggml's AVX2 i-quant kernels, which the copy uses on the CPU, build every 4-8 weights from scalar lookups in the grid and sign tables. They are limited by instruction count, not RAM: Strata measured about 5 GB/s per core even on Zen 4. The rate on this core at its sustained clock is *S1*. The copy keeps these kernels: if the CPU is the limit, a larger share of each layer's misses goes over DMA instead (S2's autotune). |
| **6 physical cores, HT siblings share ports, nothing else runs** | 6 compute threads are pinned to physical cores. The doorbell spinner (`pause` loop) runs on an HT sibling. The OS, IRQs and server use the other siblings. The sibling numbering is *S1*. |
| **DDR4-2667 dual channel, 42.7 GB/s theoretical, shared by CPU reads and GPU DMA** | The CPU's miss rate plus the DMA rate is capped at the usable RAM bandwidth. That cap, alone and with DMA running, is *S1*. |
| **16 GB RAM; headless Linux uses about 1.2 GB** (*S1*) | The whole 3.06 bpw expert set (11 GiB) sits pinned in one arena. The copy already uses 2 MB huge pages (`MAP_HUGE_2MB`, or THP as fallback). 1 GB pages (`pdpe1gb`, 16 STLB entries) are a small, measured extra (S1 step 6, applied in S2). |
| **WD SN720 NVMe** | Read once at start; after that unused in all-pinned mode. The load time is *S1*. |
| **Thin chassis, shared heat pipes, CPU 45 W + GPU ~80 W** (TGP is *S1*) | Decode needs memory clock, not GPU core clock. A lower GPU clock lock may give the CPU thermal headroom. Whether `-lgc` works on this card, and what it gains, is *S1*, applied in S2. |
| **Linux** | The box is the only target. The dev PC stays Windows and needs no OS changes: the binaries are built in a container, and the proxy runs in the WSL2 Ubuntu that is already installed (phase 0). |

### 2b. The OS, set up for one job (applied in S2)

Every item is a reversible boot or service setting. Each one is measured on and off once and kept only if it pays.

- **Power profile:** the platform profile or fan mode set to performance, if Linux exposes one (S1 checks).
- **Kernel command line:**
  - `isolcpus=managed_irq,domain,<compute cores> nohz_full=<…> rcu_nocbs=<…> irqaffinity=<siblings>`. The CPU lists
    come from S1's topology. `nohz_full` needs `CONFIG_NO_HZ_FULL`, which S1 checks.
  - `pcie_aspm=off`, so doorbells and small DMAs never wait for the link to wake.
  - If 1 GB pages win: `default_hugepagesz=1G hugepagesz=1G hugepages=12`. This leaves about 3.5 GiB for
    everything else, and the model then loads through a small page cache.
- **Services:** `multi-user.target`, swap off, the `performance` governor, `nvidia-smi -pm 1`.
- **GPU mode:** compute mode `EXCLUSIVE_PROCESS`, after checking that only the engine opens a CUDA context. The
  server's GPU query uses `cuInit` and device attributes only, and that's all it may do.
- **Engine process:** a systemd unit with `LimitMEMLOCK=infinity`, and real-time priority only for the doorbell
  thread.

## 3. The model

From config.json and a header parse of every byteshape file:

- **Layers:** 40. 30 are Gated DeltaNet (16 K heads and 32 V heads x 128, conv 4). 10 are full attention, every
  4th layer (3, 7 … 39): 16 Q heads, 2 KV heads, head_dim 256, output gate, partial mRoPE (64 of 256 dims,
  θ = 1e7).
- **Experts:** 256 per layer, 8 used per token, plus 1 sigmoid-gated shared expert. The main layers store gate and
  up as one fused `ffn_gate_up_exps` tensor.
- **Other:** vocab 248,320; untied embeddings; about 2.95 B parameters active per token.
- **MTP layer:** blk.40, 567 MiB. Its 528 MiB of experts are Q5_K, and its attention K/V, shared expert and router
  are BF16.
- **Formats in the files:** IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K, Q8_0, BF16, F32. The "IQ3_S" and "IQ4_XS" in the
  names are byteshape's display labels only.

| File | Experts | Dense / token | Expert MiB / token | RAM, all pinned / low-RAM* | Spillway mid, no MTP / MTP | |
|---|---:|---:|---:|---:|---:|---|
| 2.25 bpw | 7.7 GiB | 870 MiB | 248 | 8.3 / 2.4 GiB | 91 / 169 † | speed fallback |
| **3.06 bpw** | 11.0 GiB | 936 MiB | 353 | **11.6** / 5.8 GiB | **78 / 115** | **target, all pinned** |
| 3.53 bpw | 12.7 GiB | 1148 MiB | 407 | 13.3 / 7.7 GiB | 66 / 92 | BF16 tensors outside MTP too |
| 3.97 bpw | 14.7 GiB | 1031 MiB | 470 | 15.2 / 9.5 GiB | 58 / 70 | quality mode, low-RAM, static cache |
| 4.19 bpw | 15.5 GiB | 1110 MiB | 496 | 16.1 / 10.4 GiB | 54 / 64 | little gain over 3.97 |

\* Of about 14.8 GiB available. † Above the all-in-VRAM bound (~145), so not credible.

**Choices we make because we can** (none of them re-quantizes anything: byteshape's files are used as published):

- **MTP layer:** keep byteshape's Q5_K experts and run them natively; ggml's Q5_K kernels are already in the copy's
  `third_party/ggml` (phase 5).
- **Draft vocabulary:** the copy already drafts on a reduced vocabulary (`--draft-vocab`), which cuts the head
  from 186 to about 30 MiB per draft. We rebuild it from our own traces. Verification uses the full head, so the
  output is exact.
- **Vision:** out of scope for v1. The 0.9 GB mmproj would take expert slots.
- **Expert-pruned variants:** no. One REAP file keeps 133 of 256 experts at 9.4 GB, loses 2.5-4.5 points on code
  benchmarks (its own card), and saves less than 2.25 bpw does.

## 4. How we project speed

`python perfmodel/speed_model.py` (add `--ctx N` or `--selftest`) treats decode as bytes and fixed costs per verify
window:

- **GPU:** dense weights, cached experts, the KV cache and the DeltaNet state, each read once per window.
- **CPU and DMA:** the misses, with CPU and DMA together capped by DDR4.
- **Fixed costs:** per layer, plus the k-1 sequential MTP drafts.
- **Hit rate:** a power law in the cached fraction.

**Calibration** (`--selftest`). Every measurement lands inside the model's lo-hi range:

| Measured | Projected |
|---|---|
| Strata paper, RTX 5070, Q2_0, 4K: 94.6 tok/s with MTP, 47-57 without, hit rate 0.72 | 83 / 48, hit 0.71 |
| llama.cpp, RTX 2070 8 GB + 32 GB DDR4, Q6: ~18 tok/s | 14 / 18 / 22 |
| llama.cpp #30059 cache, RTX 3070 Laptop 8 GB, this model IQ2_M: 35.0 -> 82.2 | 36-70 -> 53-98 |
| Strata #1402, RTX 4070 Ti sized as 8 GB, UD-IQ3_S + MTP: 70-89 | 55-124 |
| byteshape, RTX 5060 Ti, this model all in VRAM, llama.cpp: 120.7 / 169.5 with MTP | 80-116 / 105-181 |

**What the review corrected** (10 Oct):
- The KV cache is now read once per window, not once per token.
- Drafting is modelled per draft instead of as a flat +8%. That matters on this small model.
- llama.cpp's per-layer costs are split, so GPU-only runs aren't charged for offload sync.
- The copy is charged a measured 17% port gap (copy_eff 0.83, fitted to #1402).
- The tuned CPU tier is no longer assumed to run at RAM speed.
- The hit curve is now a range: f^0.55 / f^0.45 / f^0.35. All three still sit below Strata's own Figure 3 (0.50 at
  f = 0.18, ~0.93 at f = 0.53).

**Largest unknowns, in order:**
1. This model's hit-rate curve. Phase 2 measures it from routing traces, on the dev PC.
2. The VRAM the copy really leaves for the cache headless (S1 plus phase 2).
3. The CPU's sustained clock and expert rate under combined load (S1).
4. The real PCIe and doorbell costs (S1).

Treat projections as ±30% until S1 and phase 2.

**Development without the box:** the dev PC stays as it is. Its WSL2 Ubuntu 26.04 sees the RTX 3080 (driver
617.14) and has 15 GiB of RAM, close to the box's 16 GB. Strata is known to run there: a community report ran
it in Docker on WSL2 with an RTX 5080, with large pinned arenas.

Under WSL2, a few things behave differently from the box:
- no hugetlb pages (the engine falls back to transparent huge pages);
- no KV streaming;
- the Windows display driver sits underneath, which adds some per-launch and sync cost;
- the 3080 also drives the Windows desktop.

So proxy numbers are relative: they rank choices and catch regressions. The box sessions supply the absolute
numbers.

- **GPU:**
  - Run `compute_75` PTX through the driver JIT with `STRATA_EMULATE_CC=75`, so the sm_75 code paths run with sm_75
    limits.
  - Profile with Nsight Compute.
  - Per-SM bandwidth is about the same on both cards (10.7 GB/s per SM on the 2070 Max-Q, 11.2 on the 3080). That
    is only a first-order argument, so S1 measures bandwidth against active SMs on the box, the same binary
    runs on the 3080, and the two curves map one to the other.
- **CPU:** `llvm-mca -mcpu=skylake` for port pressure; the 5700X at 6 pinned threads, clock-capped to the box's
  S1 all-core MHz. S1's CPU numbers come from the real chip, so no cloud VM is needed.
- **Correctness:** per-kernel parity, plus teacher-forced KL and greedy agreement against llama.cpp on the same GGUF.

## 5. Box sessions: the only scarce resource

Development happens on the dev PC with agents, and it is fast. **Box time is what we budget.** There are two
sessions, and each one is:

- **one command, supervised:** someone starts it and watches; a step that goes wrong is simply rerun;
- **self-logging:** a 1 Hz sensor logger runs for the whole session (per-core MHz, package W, temperatures, GPU
  clocks, W, °C and throttle reasons), so every benchmark also records the machine's thermal state;
- **a single archive:** it ends with `scout-<date>.tar.gz` / `deploy-<date>.tar.gz`, which goes back to the dev PC.

**One folder, one command:** copy `out/s1kit/` to the box and run `bash s1kit/s1.sh`. It checks the driver,
downloads the model into the folder if it isn't there yet (not session time), then runs S1.

**If the box stays reachable over SSH,** S1 and S2 can happen in one access window. Development runs between
them while the box idles.

### S1: scouting — at most 10 minutes, no root needed, no changes

| Step | Budget | What | Settles |
|---|---:|---|---|
| 1 | 20 s | Inventory: `perfmodel/probe.sh` (OS, kernel config, THP, huge pages, CPU topology and siblings, governor/turbo, RAPL limits, `nvidia-smi -q`, PCIe link and ASPM, platform profile, disk, toolchain, network). Serial numbers and UUIDs are filtered out. | which CUDA toolkit to build with (CUDA 13 needs a 580+ driver; CUDA 12.x runs on 525+), the 2b settings, the model's constants |
| 2 | 40 s | Micro-benchmarks, one small C/CUDA binary: CUDA context size and free VRAM; GPU read bandwidth at 1-36 active SMs; H2D/D2H pinned and pageable at 64 KB-64 MB, including 1.2 MB expert-sized copies; doorbell round trip through mapped memory; CUDA graph launch cost per node; DDR4 read bandwidth at 1/6/12 threads, alone and during an H2D stream | `gpu_eff` and the 3080-to-box mapping, `pcie_bw`/`pcie_eff`, `layer_us`, `ram_eff`, the shared-DDR4 cap, the VRAM budget |
| 3 | 45 s | CPU expert kernels: `scout cpuq` calls ggml's own `vec_dot` for IQ2_XXS / IQ3_XXS / IQ4_XS, on 2048- and 512-column rows (gate_up and down), at 1 and 6 threads pinned to physical cores. These are the same ggml kernels the copy uses. | `cpu_core_gbs`; the CPU/DMA split of each layer's misses |
| 4 | 5 min | **The real baseline, which is also the thermal soak:** `llama-batched-bench` on the 3.06 bpw file (`llama-bench` has no `--moe-cache-mib`): pp2048 and pp8192, each followed by tg128, in two configurations, both sized from step 2's free VRAM (batched-bench skips `--fit`): `--n-cpu-moe` with as many layers' experts on the GPU as fit, and `-cmoe --moe-cache-mib` sized from step 2's free VRAM, stepping down on out-of-memory. | the gate's denominator; validates two of the model's four engines on the real machine |
| 5 | 2 min | Combined synthetic load: the GPU bandwidth loop plus a 6-thread CPU expert loop. If passwordless `sudo` exists, the last 60 s run with the GPU core clock locked low (`nvidia-smi -lgc`, undone with `-rgc` before the step ends). | sustained clocks under the real load mix; whether a GPU clock lock buys CPU headroom |
| 6 | 20 s | Only with passwordless `sudo`: reserve 2 MB and 1 GB pages at runtime, pin each with `cudaHostRegister`, measure H2D and CPU streaming against 4K pages, then release them. | whether 1 GB pages work and pay |
| 7 | 5 s | Pack the logs and results into the archive. | |

**Dropped from S1, because something else already gives the answer:**
- a separate disk test (step 4's model load times the read);
- llama.cpp with MTP (with experts offloaded it gains about 2% per community runs, and our MTP path is the copy's);
- a longer soak (if step 4-5's log still shows clocks falling at the end, S2's validation, which runs longer under
  load, settles it).

**The kit** (`kit/`): `llama-batched-bench`, `nvidia-smi`, `turbostat` if present, one C/CUDA micro-benchmark
binary (`kit/src/scout.cu`) and the runner (`kit/s1.sh`).

- **Build:** `kit/build.sh` in the `kit/Dockerfile` container (Ubuntu 20.04, CUDA 12.8, sm_75 + sm_86). Static
  libstdc++, bundled CUDA libraries: it needs only glibc 2.31+ and a 525+ driver.
- **Dry run:** the whole kit runs on the dev PC within budget before it goes to the box.

### S2: deployment and on-device tuning — at most 20 minutes, needs root, one reboot

| Step | Budget | What |
|---|---:|---|
| 1 | 30 s | Install the engine, pack, config, systemd unit and an **undo script** that restores every file and setting it changes. |
| 2 | 2 min | Apply section 2b (kernel command line, governor, persistence and compute mode), then **reboot once, early**. Everything after runs on the final OS. A one-shot systemd unit resumes the run after the reboot. |
| 3 | 8 min | **Autotune** of the choices only the box can settle. These are pre-ranked on the proxy. Each trial is a short decode benchmark, and the tuner keeps the best so far when the budget runs out. The choices: CPU worker count and doorbell placement, DMA miss share (auto against fixed), the 36-SM launch-shape variants, `MMVQ_IL` variants, the expert-cache margin, MTP draft length k = 2-4, and the GPU clock lock (if S1 showed it helps). |
| 4 | 6 min | **Validation with the winning config:** the S1 workloads (8K and 32K, MTP on), then sustained decode until the budget ends. This doubles as the long thermal check. |
| 5 | 10 s | Archive the results. The service stays running with the winning config. |

## 6. Phases and gates

| # | Phase | Where | Output |
|---|---|---|---|
| 0 | Build setup, with nothing installed on Windows: Linux binaries are built in a CUDA devel container on an old Ubuntu base (Docker Desktop, already installed), so they run on the box's distro whatever it is; the S1 kit is built and dry-run in WSL2 within its budget | dev PC | the kit; a WSL2 proxy |
| S1 | Skipped: with supervised box access we build first and measure on the box. `out/s1kit/` stays as the llama.cpp baseline for S2 (`bash s1kit/s1.sh`) | | |
| 1 | Copy, strip, load byteshape's file, parity | dev PC | `engine/` builds for 75;86 and passes parity on the 3.06 bpw file |
| 2 | Proxy baseline at the box's budget (8 GB minus the CUDA context) | dev PC | the copy's tok/s and slot count; this model's hit-rate curve; refit |
| | **Gate G1:** on the proxy, the copy must reach at least 1.5x llama.cpp `--moe-cache-mib` at the same VRAM budget (llama.cpp: 73-89 tok/s on the 3080 in the dry run). Otherwise deploy llama.cpp in S2 and stop. | | |
| 3 | VRAM accounting | dev PC | at least 45% of experts cached at 32K context |
| 4 | Turing kernels, launch fusion, switch sweep | dev PC | the sm_75 work from four PRs; ranked candidates for S2's autotune |
| 5 | MTP for this file | dev PC | the Q5_K MTP experts running natively, a draft vocab, ranked k |
| 6 | Server cut to the OpenAI API; config; the S2 kit (installer, undo, resume, autotune) | dev PC | the S2 kit, dry-run on the proxy |
| S2 | **Deployment and on-device tuning** (≤ 20 min) | box | the final config running; the final number |

### Phase 1: copy and strip

**The snapshot:**
- `650cf81` (the head of #1820) goes into `engine/` as one commit in our repo.
- `engine/SOURCE.md` records the SHA.
- Keep `LICENSE` and `third_party/ggml`'s license (MIT's one condition).

**Byteshape compatibility:**
- **Fused gate/up:** the packer (`tools/iq_pack.py`) expects separate `ffn_{gate,up,down}_exps`. Byteshape fuses
  gate and up into `ffn_gate_up_exps`, so the packer must split it, with the row order checked against llama.cpp's
  loader.
- **MTP experts:** they are Q5_K, which #1402 may not run natively. Until phase 5 wires in ggml's Q5_K kernels,
  MTP is off or uses the F16 loader.
- **Main experts:** IQ2_XXS down projections on 15 layers of the 3.06 file. The copy has CUDA and AVX2 kernels for
  IQ2_XXS. Check that the packer accepts them.

**Parity:** teacher-forced KL and greedy agreement against llama.cpp on the same file. The repo's `*_parity` tests
are the template.

**Strip in steps, building and running parity after each one:**
- the Flash-Next-only paths (QSA sparse attention, PLE n-gram table, GR hyper-connections, Q2_0 S2 packs, Flash-Next
  setup);
- `sycl/`, HIP, Windows-only code and multi-GPU;
- vision, `setup.py`'s installer, the MCP server, Docker and the translated READMEs.

### Phase 2: proxy baseline

**Make the RTX 3080 look like the box:**
- `compute_75` PTX with `STRATA_EMULATE_CC=75`;
- a filler process that holds exactly enough VRAM to leave the box's S1-measured free amount, after what the
  Windows desktop already uses (#1402 measured its "8 GB" rows the same way);
- 6 threads pinned, and the CPU clock capped to S1's sustained all-core MHz;
- PCIe stays at the 3080's Gen4: the model scales the PCIe terms to S1's measured link instead of a BIOS change.

**Measure:**
- the copy against llama.cpp, at 1K, 8K and 32K, MTP on and off;
- the copy's expert slot count;
- whether low-RAM mode adapts its VRAM set;
- routing traces with `--dump-routing` over a few hundred prompts (code, chat, German), which give the hit-rate
  curve, plus our own expert profile via `tools/make_profile.py`.

### Phase 3: VRAM accounting (the largest modelled gain)

Account for every allocation at start-up, then cut. The suspects:
- #1820's fixed 1 GiB haircut and slack;
- prompt buffers not borrowed from the cache;
- cuBLAS workspaces and graph memory;
- the full MTP layer (567 MiB, phase 5 shrinks it);
- BF16-to-FP32 expansions in the compat pack;
- KV or scratch sized for Flash-Next's shapes.

Each cut is recorded with a before/after slot count and a parity run.

### Phase 4: Turing kernels

**Bring in, and wire to `qwen35moe` shapes:**
- #1581 (from an RTX 2080 Super Max-Q 8 GB laptop: SM75 prompt attention, native-IQ experts, prompt VRAM loans);
- #1821 and #1825 (int8 KV prompt attention at 29.2 KB shared memory; bit-exact);
- #1437 (`STRATA_MMVQ_IL`, interleaved MMVQ for verify windows).

The commit SHAs in section 7 come from the fork survey. Check that each one applies to `650cf81`.

**Then the copy itself:**
- Count launches per layer; fuse toward 6.
- Check the DeltaNet kernel's state traffic.
- Sweep every `STRATA_*` switch on the proxy. Clear winners become hard defaults and losers are deleted.
  Choices the 3080 can't settle (36-SM shapes, `MMVQ_IL`) go to S2's autotune as ranked candidates.

### Phase 5: MTP

- Run the Q5_K MTP experts natively.
- Rebuild the draft vocabulary.
- Rank `--spec` 2-4 on the proxy; S2's autotune makes the final pick.

### Phase 6: the S2 kit

- An installer and an undo script.
- The section 2b settings.
- A resume-after-reboot unit.
- The autotune runner with its ranked candidates and budget.
- 1 GB pages, only if S1 step 6 showed a gain: `MAP_HUGE_1GB` instead of `MAP_HUGE_2MB` in `src/core/pinned.cu`,
  which already pins its hugetlb mapping with `cudaHostRegister`.

Dry-run the whole kit on the proxy, including the reboot path, before S2.

## 7. What we copy from where

| Source | Taken |
|---|---|
| Strata @ `650cf81` (#1820; contains `fb58e0d` and #1402 @ `48534f3`) | the whole engine, `qwen35moe` support, F16 MTP loader, VRAM cap, chunked DeltaNet prompt, Qwen3.6 expert profile, `STRATA_EMULATE_CC` |
| #1581 (`0a94be3`, `0b51978`, `e9b4f47`), #1821 (`74616ca`), #1825 (`14bf321`), #1437 (`d006768`) | Turing kernels (phase 4) |
| llama.cpp | nothing; the parity oracle and the baseline to beat |
| lighttransport `tools/sim/` | nothing; a second opinion on hit-curve shape (h = 1.16·f^0.545 on GLM) |

**Not taken:**
- **Other forks:** most of the ~1,800 forks are untouched mirrors. Of the renamed ones you named, five sit at
  upstream HEAD; Strata-for-p100 only adds CI and a Dockerfile, and Strata-ru only changes the README.
- **`data/experimental-speed-projection`:** a refusal-removal vector for Qwen3.8 (its own README), not a speed tool.
- **Everything phase 1 strips.**

## 8. Files here

- `perfmodel/speed_model.py`: the speed model (4 engines x 5 files, MTP on/off, `--ctx`, `--selftest` with six
  calibration checks).
- `perfmodel/probe.sh`: S1 step 1, the read-only inventory (within its 20 s budget).
- `PLAN.md`: this plan.
