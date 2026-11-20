# Isolated Speculative Sandbox Worker Operations

This directory contains operator-reviewed templates for deploying the isolated sandbox execution worker on a dedicated Linux node. These scripts are intended for explicit manual administration and are never invoked by models or auto-deploy pipelines.

## Security & Credential Boundaries

- **Zero provider keys on the worker**: LLM provider API keys stay strictly on the control-plane host. The worker only receives an independent random `SPEC_WORKER_TOKEN` (>= 32 characters) for authenticating its outbound reverse WSS connection.
- **No inbound ports on the worker**: The worker dials out to `wss://<control-plane>/api/spec-worker/ws` over verified TLS (never disable certificate verification). On the control-plane Nginx reverse proxy, restrict `/api/spec-worker/ws` to the worker's IP (see `nginx-spec-worker.conf.example`).
- **Private repositories via GitHub App**: Configure `SPEC_GITHUB_APP_*` only on the control plane. Short-lived, single-repository, read-only installation tokens are transmitted in-memory over authenticated WSS and supplied to Git via an ephemeral `GIT_ASKPASS` helper (`SPEC_GIT_USERNAME` / `SPEC_GIT_PASSWORD`). Tokens are never embedded in URLs, written to `.git/config`, or persisted in the control-plane outbox or worker journal.

## Host Requirements

- Linux with cgroups v2, unprivileged user namespaces, and `bubblewrap >= 0.11` (supporting `--overlay-src` and `--tmp-overlay`).
- System runtimes installed under `/usr`: `git`, `bubblewrap`, Python 3.11+, and any project runtimes (`node`, `npm`, `pnpm`, `uv`, etc.).
- Verify bubblewrap as the dedicated `spec-worker` user before enabling the service:
  ```sh
  sudo -u spec-worker bwrap --unshare-all \
    --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
    --proc /proc --dev /dev -- /usr/bin/true
  ```
  If this fails, fix host userns settings—the worker intentionally refuses to fall back to unsandboxed execution.

## Disk Quota & Service Setup

1. Provision a dedicated 30 GiB filesystem (either a dedicated block volume or by reviewing and running `SPEC_PROVISION_DISK=1 ./provision-disk.sh` on a host with >= 38 GiB free). Dedicated mounts use `O(1)` `statvfs` disk accounting and `X-fstrim.notrim` to preserve loop allocations.
2. Install this package into `/opt/spec-worker/.venv`:
   ```sh
   python3 -m venv /opt/spec-worker/.venv
   /opt/spec-worker/.venv/bin/pip install /path/to/speculative-coding
   ```
3. Copy `worker.env.example` to `/etc/spec-worker.env` (`root:root`, `chmod 600`) and set `SPEC_WORKER_TOKEN` and `SPEC_WORKER_SERVER`.
4. Copy `spec-worker.service` to `/etc/systemd/system/spec-worker.service`, then run:
   ```sh
   systemctl daemon-reload
   systemctl enable --now spec-worker.service
   ```

## Dependency Baselines & Execution Isolation

- **Golden dependency baselines**: When `SPEC_AUTO_SEED_DEPS=1` (default), the worker computes a manifest digest (`pnpm-lock.yaml`, `package-lock.json`, `uv.lock`, `requirements.txt`, `pyproject.toml`), tries offline installation from `/opt/spec_sandbox/shared_caches` first, and on cache miss runs a controlled `--ignore-scripts` network-enabled `bwrap` install to seed the shared cache and immutable read-only baseline. Set `SPEC_AUTO_SEED_DEPS=0` for strict air-gapped operation.
- **Offline candidate execution**: All candidate `command`, `gate`, and `preview` steps execute inside `bwrap --unshare-all` with zero network access, read-only golden dependencies, copy-on-write tmpfs overlay over `shared_caches`, and process-group timeout/backpressure cleanup.
- **Ref-only dehydration**: Nominated candidates are committed to `refs/spec/<cand_id>` in the bare repository, their diff/report/preview artifacts (<= 13 MiB raw / < 20 MiB packaged) are returned to the control plane, and the hydrated worktree directory is immediately removed.
- **Human-only merge**: Models only have `DROP`, `REVISE`, and `NOMINATE` verdicts. Merging requires explicit human confirmation with reviewed `approved_target_sha` and `approved_head_sha`, uses `git merge-tree --write-tree` + CAS `update-ref`, and never pushes upstream.
