# guardian-server

llama.cpp server with the Guardian shared-memory front-end, for the
[Guardian](https://github.com/TUM-DSE/guardian) confidential-computing runtime.

Same binary surface as `llama-server` for the core HTTP endpoints (`/health`,
`/metrics`, `/completion(s)`, `/chat/completions`, `/(v1/)embeddings`,
tokenize/detokenize, slots), plus a shared-memory request channel speaking the
Guardian shm protocol:

- Region: POSIX shm `Guardian_shm` by default; `$GUARDIAN_SHM_PATH` overrides
  with a plain file path (e.g. an ivshmem PCI BAR), `$GUARDIAN_SHM_NAME`
  overrides the shm name. Layout is the canonical `guardian/shm.h` from the
  guardian repo's `native/` tree (512 slots × 8 MiB text field; ABI pinned by
  `native/tests/abi_check.cpp`).
- Handshake per slot: client takes a free slot mutex, writes the request JSON
  (same body as `/completion`: `prompt`, `n_predict`, … plus `prio`,
  `outb_id`), posts `serverNotifier` + `active_reqs`; the server answers into
  the slot's text field (llama.cpp native completion JSON incl. `timings`) and
  posts `clientNotifier`.
- Priority: legacy `prio` 0–9 (lower = higher); `prio <= 1` maps to
  front-of-queue posting.
- Benchmark instrumentation: writes `(outb_id << 16) | event` to I/O port
  0xf4 (event 201 = request pickup, 202 = result posted) when `ioperm`
  succeeds (root); silently disabled otherwise.

Build (from the repo root; requires the guardian repo checkout for the ABI
header — path configurable via `-DGUARDIAN_NATIVE_DIR`):

```
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-guardian-server
build/bin/guardian-server -m model.gguf --host 0.0.0.0 --port 8080
```

This tool replaces the legacy `examples/guardian_server` from the old
TUM-DSE fork (branch `guardian-legacy`), reimplemented against the
modularized `server-context` API (`server_response_reader`).
