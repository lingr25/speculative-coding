# Speculative Coding Architecture

## Overview

Speculative Coding is an autonomous exploration and code-evolution engine designed around **dual-node separation**, **strict provider pool isolation**, **deterministic verification gates**, **ref-only dehydration**, and **human-only merge authority**.

Instead of waiting for explicit instructions, Speculative Coding uses surplus/low-cost LLM capacity (isolated in a dedicated `speculative` provider pool) to continuously inspect repositories, branch multiple candidate directions, implement and verify them inside isolated sandboxes, strictly review code quality and runtime completeness, and surface only high-signal nominated candidates for human review.

## Core Pipeline

```text
[Control Plane: SpeculativeOrchestrator]
   │
   ▼
1. Recon ───────────────► Inspect bare repo tree, recent commits, memories, and graveyard.
   │                      Produce 0..N high-value routes (or sleep quietly if none).
   ▼
2. Fan-Out & Hydrate ───► Dispatch worktree creation to isolated Sandbox Worker.
   │                      Parallel worktrees share immutable read-only dependency baselines.
   ▼
3. Code & Gate Loop ────► Coder agent uses parallel spec_grep/spec_read + surgical spec_edit/spec_write.
   │                      Deterministic gates (build/test/lint) + optional acceptance preview run offline.
   ▼
4. Strict Review ───────► Independent Reviewer (+ Playability/Completeness Inspector for feature routes).
   │                      Verdicts: DROP (tombstone in graveyard), REVISE (loop back), NOMINATE.
   │                      Stagnant diffs (2 identical rounds) or max rounds -> automatic DROP.
   ▼
5. Dehydrate & Stage ───► Commit to refs/spec/<cand_id>, return concise report + diff + preview bundle,
   │                      and immediately delete the physical worktree directory.
   ▼
6. TTL & Human Verdict ─► Default TTL auto-expires unreviewed candidates.
                          Human actions: Keep (freeze TTL), Steer (re-hydrate & revise),
                          Drop (tombstone), Export (patch), or explicit SHA-pinned Merge (no push).
```

## Dual-Node Separation

| Layer | Responsibility | Credentials & Isolation |
| :--- | :--- | :--- |
| **Control Plane** (`orchestrator.py`, `agent.py`, `automation.py`, `server.py`) | Async LLM orchestration, SQLite metadata (`spec_*`), conversational delegation agent, scheduled reviews, HTTP/WS API. | Holds LLM API keys & optional GitHub App key. Zero local build subprocesses or heavy dependency installs. |
| **Sandbox Execution Worker** (`worker.py`, `sandbox.py`) | Bare Git repos, detached worktrees, read-only golden dependency baselines, offline `bwrap --unshare-all` gates, dehydration & GC. | Zero LLM API keys. Connects outbound via reverse WSS with `SPEC_WORKER_TOKEN`. Runs under `systemd` memory/CPU cgroup limits and 30 GiB disk budget. |

## Concurrency & Resource Guardrails

1. **Three-tier concurrency**:
   - L1 LLM inference slots (`llm_concurrency`, default 6, global cap 8) on the control plane.
   - L2 worktree slots (`wt_concurrency`, default 3, max 4) on the worker.
   - L3 heavy build/gate slots (`build_concurrency`, default 2, throttled to 1 under high disk pressure).
2. **Immutable golden dependency baselines**: Lockfile-hashed `node_modules` / `.venv` directories are built once and mounted `--ro-bind` into candidate sandboxes so parallel worktrees never duplicate gigabytes of dependencies or mutate shared state.
3. **Three-level disk pressure & LRU eviction**: Normal (< 75%), High (75%–100%, evicts unleased `golden_bases` LRU and serializes builds), Blocked (>= 100% or host free < 8 GiB, watchdog kills runaway commands and pauses admission).
4. **Crash & reconnect resilience**:
   - Read-only queries (`read`, `grep`, `diff`, `describe`, `status`) use an ephemeral fast path bypassing SQLite journals.
   - Mutations (`dehydrate`, `gate`, `command`, `merge`, `drop`) are journaled with deterministic task IDs and payload digests on both control plane (`spec_worker_tasks`) and worker (`task-journal.sqlite3`), guaranteeing idempotent replay across disconnects and restarts.
5. **Security & capability boundaries**:
   - Models never receive merge, push, deploy, credential, or TTL-control tools.
   - Protected paths (`.git`, `.env*`, `.ssh`, `deploy`, `.github`, `AGENTS.md`) and existing tests (`tests/*`, `test_*`) cannot be modified or deleted by candidates; adding new test files is permitted.
   - Preview artifacts are served with strict sandboxed CSP (`sandbox allow-scripts; connect-src 'none'`).
   - Bidirectional LLM pool isolation ensures `speculative` inference never falls back to paid `main` providers and sleeps cleanly when all speculative endpoints are cooling.
