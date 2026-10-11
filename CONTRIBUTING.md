# Contributing

Spillway targets one machine: the RTX 2070 Max-Q laptop (Turing, 8 GB, 16 GB RAM). Changes that speed it up, fix it or
make it easier to run are welcome. Support for other hardware is out of scope.

- Build and test: [docs/DEVELOPING.md](docs/DEVELOPING.md). `./dev.sh test` must pass.
- Engine changes: measure them on a Turing card with `dev/release_ab.sh` or `dev/ab_fetch.sh`, check parity (identical
  tokens, or the KL against the old path), and add the numbers to `docs/BENCHMARKS.md` and the change to
  `engine/SOURCE.md`.
- A new engine setting that the laptop has to decide goes into the tuning sweep in `kit/install.sh`.
- Bugs: open an issue with the output of `spillway report`.
