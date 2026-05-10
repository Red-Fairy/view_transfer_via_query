# Changelog

## 2026-05-09
- `scripts/_common.sh`: export `LIBRARY_PATH`, `LD_LIBRARY_PATH`, `CPATH` to include `/opt/xpmem/lib64` and `/opt/xpmem/include` when that directory exists.
  - **Why:** On DeltaAI (gh-nodes / Cray + GH200), the `xpmem` module only populates runtime `LD_LIBRARY_PATH`. DeepSpeed JIT-compiles ops that pull in cray-mpich → `-lxpmem`, and the linker fails with `cannot find -lxpmem` because compile-time `LIBRARY_PATH` is empty.
  - **Scope:** guarded by `[ -d /opt/xpmem/lib64 ]`, so it's a no-op on non-Cray machines.
