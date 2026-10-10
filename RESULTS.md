# Measured results

Proxy = the dev PC's RTX 3080 10 GB in WSL2 (Windows desktop on the same card), Ryzen 7 5700X, 3.06 bpw file,
greedy, 4K context, `perfmodel/bench_engine.sh` (one coding prompt, 128 tokens) unless noted.

## llama.cpp (781dbc5), `out/s1kit` dry run, 3080 proxy

| Config | pp2048 | tg after 2K | pp8192 | tg after 8K |
|---|---:|---:|---:|---:|
| `--n-cpu-moe 15` (as many layers' experts on the GPU as fit) | 860 | 33.3 | 1135 | 36.7 |
| `-cmoe --moe-cache-mib 7367` | 584 | 72.8 | 956 | 89.2 |

## Engine copy (650cf81 + byteshape fixes), 3080 proxy

| Date | Change | Decode tok/s | ms/window | Notes |
|---|---|---:|---:|---|
| 10 Oct | baseline, MTP off, `--spec 2` | 76.8 | 13.2 | profile on; cache 5,465 slots (5.9 GiB). GPU: VRAM-hit experts 5.1 ms/window (~63 GB/s effective) |
| 10 Oct | expert layout mode 8 default (was 0 on non-Volta) | 114.0 | — | profile on; VRAM hits 5.1 -> 1.7 ms/window. Sweep: mode 0 72.9, 2 86.0, 5 103.6, 6 113.3, 7 110.3, 8 114.0 (1, 4 did not run). Greedy output identical to mode 0 over 128 tokens (fixed cache 4000) |
| 10 Oct | BF16 MMVQ for byteshape's MTP layer (draft K/V + shared expert), MTP on | **148.4** | 18.2 | `--spec 3 --spec-min-p 0.5`, `data/draft_vocab.bin`; 2.69 tokens/window, drafts 162/184 accepted. spec 2: 137.7, spec 4: 132.2; no MTP: 92.6 (256 tokens, profile off) |

## Parity (phase 1), greedy vs llama.cpp `llama-completion` on the same GGUF, 64 tokens (`perfmodel/parity_greedy.py`)

| Prompt | Engine, MTP off | Engine, MTP k=3 |
|---|---|---|
| code (merge sorted lists) | identical | identical |
| chat (stay focused) | split at 46: "specific" vs "certain" | identical |
| German (Kühlschrank) | split at 28: "um die Temperatur zu" vs "wodurch sich die Temperatur" | split at 29 |

The splits are near-tie wordings, the IQ3-level rounding upstream documents for this model (docs/QWEN36.md).

## Box budget on the proxy (phases 2-3): VRAM held to 7,800 MiB free (`scout hold`), `STRATA_EMULATE_CC=75`, 6 cores (`taskset -c 0-11`, `--pool-workers 5`), MTP k=3

| Expert cache | Decode tok/s | ms/window | Note |
|---|---:|---:|---|
| auto | — | — | refuses: the WDDM-only post-touch haircut (`under_wddm()`, WSL counts) reads 0 MiB free; native Linux skips it |
| 6,118 slots (auto's pre-touch size, 700 MiB reserve) | 55.0 | 48.5 | WDDM spilled VRAM to system RAM (GPU wait 40 ms): proxy artifact |
| 5,400 slots (53% of the 10,240 experts) | **142.8** | 18.9 | 2.69 tokens/window; GPU wait 10.9, CPU pool 4.6, draft 1.3 ms/window |

The real slot count and the reserve are settled on the box (S2 autotune: `--vram-reserve-mib`).

## Phase 4: Turing PRs

| PR | Taken? | Why | Check |
|---|---|---|---|
| #1821 int8 K/V prompt attention (Turing) | yes, 3-way merge | self-contained prompt kernel | `qsa_prompt_attn_parity` PASS, also with `STRATA_EMULATE_CC=75` |
| #1825 4-lane prompt scorer (Turing) | yes, 3-way merge | self-contained | `qsa_scores_parity` bitwise identical (native and EMULATE_CC=75) |
| #1437 sm_75 interleaved verify projections | no | its table holds only Flash-Next shapes (2560/6144 wide); Qwen3.6's 2048-wide projections fall back, so no effect; 650cf81 already has the per-arch IL tables | — |
| #1581 SM75 prompt / VRAM loans | no | predates #1820 and restructures the MTP expert load for experts.bin; 12 conflicts with #1820's qwen35moe in-file draft layer; prompt-only gains | — |

After the merges: `native_grouped_parity` 0 failures (IQ3_XXS/IQ2_XXS bitwise), `mmvq_multi_parity` ok; decode 160.3 / 142.8 tok/s in two runs (MTP k=3, run-to-run spread on the WSL proxy ~±6%).

## S2 kit dry run on the proxy (`bash s2kit/s2.sh`, 10 Oct)

Install + pack 10 s; tuning 9 engine starts (~15 s each); service up and answering through the OpenAI API.

| Candidate | tok/s |
|---|---:|
| spec 2 / 3 / 4 | 155.8 / **167.1** / 159.1 |
| spec 3, VRAM reserve 450 / 250 MiB | 145.5 / 131.1 (bigger cache spills under WDDM on the proxy; the box decides) |
| spec 3, 5 / 4 pool workers | 166.9 / 161.6 |
| spec 3, expert layout 6 / 7 | 133.7 / 155.4 |

Through the service: 160.8 tok/s decode, drafts 227/270. `--undo` removed the service and /opt/strata.
