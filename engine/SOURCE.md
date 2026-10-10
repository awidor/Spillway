# Source

A copy of [Strata](https://github.com/Niko1221/Strata) (MIT, see `LICENSE`) at `650cf81`, the head of
[#1820](https://github.com/Niko1221/Strata/pull/1820), which contains upstream `main` @ `fb58e0d` and
[#1402](https://github.com/Niko1221/Strata/pull/1402) @ `48534f3` (the Qwen3.6 / `qwen35moe` port). Copied 10 Oct 2026.
Not a fork: nothing here goes upstream. `README.md` and `docs/` are upstream's and describe upstream.

## Changes against 650cf81

| Change | Files |
|---|---|
| byteshape's fused `ffn_gate_up_exps` read in place (gate rows then up rows per expert, the blob's own order) | `tools/iq_pack.py`, `include/strata/kernels/cpu/expert_layout.hpp`, `src/core/expert_source.cpp` |
| IQ2_XXS down projections in the GPU expert kernels | `src/kernels/cuda/iq_kernels.cu`, `src/kernels/native_grouped_parity.cpp` |
| Expert layout mode 8 is the default on every card (was Volta only) | `src/kernels/cuda/iq_kernels.cu` |
| BF16 weights in native MMVQ, for byteshape's MTP layer (draft K/V, shared expert); the main model's dense path unchanged | `src/kernels/cuda/native_mmvq.cu`, `src/core/native_dense.cpp` |
| Missing `<cstdint>` include | `include/strata/core/vram_floor.hpp` |
| Merged: #1821 (Turing int8 K/V prompt attention), #1825 (Turing 4-lane prompt scorer) | `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/cuda/qsa_select.cu`, their parity tests, `CMakeLists.txt` |
| Removed: SYCL engine, installer (`setup.py` and its tests), MCP server tool, Docker files, Windows launchers, translated READMEs | |
| Line endings normalized to LF | all text files |

Measurements for each change: `../RESULTS.md`.
