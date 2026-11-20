# Speculative Coding

An autonomous, dual-node speculative coding engine with an async control plane, an isolated Linux `bubblewrap` sandbox worker, a conversational delegation agent, and durable scheduled briefings.

## Highlights

- **Dual-Node Separation**: The control plane runs pure async LLM orchestration and holds provider keys; the isolated sandbox worker connects outbound via reverse WSS (`wss://`), holds zero LLM credentials, and runs all builds/tests inside `bwrap --unshare-all` with zero network access.
- **Strict Provider Pool Isolation**: `ProviderRouter` supports both OpenAI-compatible and Gemini native endpoints with multi-key failover, exponential backoff cooldowns, and strict separation between `speculative` and `main` provider pools (never falling back to paid main models when speculative providers are cooling).
- **Parallel Tool-Calling Agent Loop**: Recon, Coder, Reviewer, and Inspector roles execute with parallel read batching (`spec_grep`, `spec_read`), surgical edits (`spec_edit`, `spec_write`), isolated sandbox commands (`spec_command`), automatic atomic context compaction, and overflow recovery.
- **Deterministic Gates & Dual Review**: Candidates must pass configured deterministic gates (and optional runtime acceptance previews for feature/gameplay routes), pass strict independent review (`DROP | REVISE | NOMINATE`), and dehydrate immediately to lightweight Git refs (`refs/spec/<cand_id>`).
- **Human-Only Merge Authority**: Models never have merge, push, or deploy tools. Human actions (`keep`, `steer`, `drop`, `export`, `merge`) are idempotent, owner-scoped, and require explicit SHA-pinned confirmation for local CAS merges (never pushing upstream).
- **Conversational Delegation & Automations**: `SpecAgent` provides natural-language repo resolution (with optional GitHub App discovery), automatic gate inference and baseline validation (`SpecProvisioner`), engineering memory, SHA-bound evidence explanation, and scheduled batch/briefing runs (`SpecAutomation`) with pluggable `NotificationSink`.

## Architecture

See [docs/design/architecture.md](docs/design/architecture.md) for the full design and [deploy/spec-worker/README.md](deploy/spec-worker/README.md) for production worker hardening.

```text
speculative_coding/
├── config.py          # SpecConfig, ProviderEndpoint, strict pool isolation
├── protocols.py       # AuthProvider & NotificationSink protocols + adapters
├── llm/               # OpenAI-compatible & Gemini transports, ProviderRouter, compaction
├── agent_loop/        # Provider-agnostic AgentLoop with parallel read batching & recovery
├── tool_result.py     # Structured ToolResult & output bounding
├── store.py           # Durable SQLite metadata store (spec_* tables)
├── sessions.py        # Lightweight SQLite SessionStore, EventBus, and TurnRunner
├── sandbox.py         # SpecSandboxBackend, RemoteWorkerSandbox, LocalDirSandbox (bwrap)
├── worker.py          # Reverse-WSS sandbox execution daemon with crash-safe journal
├── github.py          # GitHub App short-lived repo-scoped token client & SpecRepoRegistry
├── provision.py       # Conservative gate inference & baseline verification
├── orchestrator.py    # SpeculativeOrchestrator lifecycle, review loop, TTL, HTTP API
├── agent.py           # SpecAgent conversational delegation layer
├── automation.py      # Durable scheduled batch runs & morning review briefs
├── server.py          # Standalone aiohttp control-plane HTTP & WebSocket server
└── cli.py             # Unified CLI (`spec-coding`)
```

## Installation

Requires Python 3.11+ and `git` on `PATH`.

```bash
pip install -e ".[dev]"
```

## Configuration

Copy [`config.example.json`](config.example.json) and configure your provider keys and tokens via environment variables:

```bash
export SPEC_OPENAI_API_KEY="sk-..."
export SPEC_API_TOKEN="your-control-plane-bearer-token"
export SPEC_WORKER_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Alternatively, for quick zero-file setup, `SpecConfig.from_env()` reads `SPEC_OPENAI_API_KEY`, `SPEC_OPENAI_BASE_URL`, `SPEC_MODEL`, `SPEC_DATA_DIR`, `SPEC_WORKER_TOKEN`, and `SPEC_API_TOKEN`.

## Usage

### 1. Start the Control-Plane Server

```bash
spec-coding serve --config config.example.json --host 127.0.0.1 --port 8765
# or:
spec-server --config config.example.json --host 127.0.0.1 --port 8765
```

HTTP & WebSocket endpoints:
- `GET /health`
- `GET /api/spec-worker/ws` (reverse WSS endpoint for the sandbox worker; requires `Bearer $SPEC_WORKER_TOKEN`)
- `GET /api/spec/status`
- `GET / POST /api/spec/projects`
- `PATCH /api/spec/projects/{project_id}`
- `POST /api/spec/projects/{project_id}/run`
- `GET /api/spec/candidates` & `GET /api/spec/candidates/{cand_id}`
- `GET /api/spec/candidates/{cand_id}/preview/{path}` (served with strict sandboxed CSP)
- `POST /api/spec/candidates/{cand_id}/{keep|steer|drop|export|merge}`
- `POST /api/spec/agent/turn` & `GET /api/spec/agent/ws` (conversational delegation agent)

### 2. Run the Isolated Sandbox Worker

On your dedicated Linux worker node (see [deploy/spec-worker/README.md](deploy/spec-worker/README.md)):

```bash
export SPEC_WORKER_TOKEN="same-32+-char-secret"
spec-coding worker --root /opt/spec_sandbox --server wss://control-plane.example.com/api/spec-worker/ws
# or:
spec-worker --root /opt/spec_sandbox --server wss://control-plane.example.com/api/spec-worker/ws
```

### 3. Local Development Batch & Inspection CLI

```bash
# Inspect stored projects, candidates, and token usage
spec-coding status --data-dir ./data
spec-coding candidates --data-dir ./data
spec-coding export <cand_id> --data-dir ./data

# Run a single local batch in dev/test mode (requires explicit --allow-local-sandbox)
spec-coding run --repo /path/to/repo --branch main --gate "python -m pytest" --allow-local-sandbox
```

## Testing

```bash
python -X utf8 -m pytest
```
