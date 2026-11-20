import {
  escapeHtml,
  formatBytes,
  formatRelativeTime,
  formatTimestamp,
  formatTokens,
  formatTtl,
  getModifiedTests,
  renderDiffViewer,
  renderEvidencePanel,
  renderMarkdown,
  shortSha,
  statusBadgeClass,
} from "./renderers.js";

/* ==========================================================================
   State & Constants
   ========================================================================== */

const TOKEN_KEY = "spec_api_token";
const THEME_KEY = "spec_ui_theme";
const ACTIVE_STATUSES = new Set([
  "coding",
  "revising",
  "building",
  "reviewing",
  "inspecting",
  "dehydrating",
]);

const state = {
  token: sessionStorage.getItem(TOKEN_KEY) || "",
  authRequired: false,
  authModalOpen: false,
  theme: localStorage.getItem(THEME_KEY) || "system",

  health: null,
  status: null,
  projects: [],
  candidates: [],

  selectedProjectId: "",
  statusFilter: "all",
  searchQuery: "",

  selectedCandId: "",
  candidateDetail: null,
  candidateEvents: [],
  activeTab: "report",
  inspectorSeq: 0,
  previewBlobUrls: [],
  activePreviewPath: "",

  // SpecAgent Chat Drawer
  drawerOpen: false,
  agentSessionKey: "",
  agentTurns: [],
  agentWs: null,
  agentWsConnected: false,
  agentRunning: false,
  agentLive: null,
};

const $ = (id) => document.getElementById(id);

function generateActionId(prefix = "act") {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return `${prefix}_${globalThis.crypto.randomUUID()}`;
  }
  const rand = Math.random().toString(16).slice(2, 10);
  return `${prefix}_${Date.now()}_${rand}`;
}

/* ==========================================================================
   Theme Management
   ========================================================================== */

const mediaDark = window.matchMedia("(prefers-color-scheme: dark)");

function applyTheme() {
  const isDark =
    state.theme === "dark" || (state.theme === "system" && mediaDark.matches);
  document.documentElement.classList.toggle("dark", isDark);
  const btn = $("btn-theme");
  if (btn) {
    btn.title = `Theme: ${state.theme} (click to switch)`;
  }
}

function cycleTheme() {
  const order = ["system", "dark", "light"];
  const next = order[(order.indexOf(state.theme) + 1) % order.length];
  state.theme = next;
  localStorage.setItem(THEME_KEY, next);
  applyTheme();
  toast(`Theme set to ${next}`, "info");
}

mediaDark.addEventListener("change", applyTheme);

/* ==========================================================================
   Toasts & Modals
   ========================================================================== */

function toast(message, kind = "info", durationMs = 3400) {
  const viewport = $("toast-viewport");
  if (!viewport) return;
  const el = document.createElement("div");
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  viewport.appendChild(el);
  window.setTimeout(() => {
    el.remove();
  }, durationMs);
}

let modalCleanup = null;

function openModal({ title, bodyHtml, footerHtml, large = false, onMount, onClose }) {
  closeModal();
  const backdrop = $("modal-backdrop");
  const dialog = $("modal-dialog");
  dialog.classList.toggle("modal-lg", Boolean(large));
  $("modal-title").textContent = title;
  $("modal-body").innerHTML = bodyHtml;
  $("modal-footer").innerHTML = footerHtml || "";
  backdrop.classList.remove("hidden");
  modalCleanup = onClose || null;
  if (typeof onMount === "function") {
    onMount({ body: $("modal-body"), footer: $("modal-footer") });
  }
}

function closeModal() {
  const backdrop = $("modal-backdrop");
  if (!backdrop || backdrop.classList.contains("hidden")) return;
  backdrop.classList.add("hidden");
  $("modal-body").innerHTML = "";
  $("modal-footer").innerHTML = "";
  if (typeof modalCleanup === "function") {
    const fn = modalCleanup;
    modalCleanup = null;
    fn();
  }
}

/* ==========================================================================
   API Client with Automatic 401 Detection -> Bearer Token Modal
   ========================================================================== */

function authHeaders(extra = {}) {
  const headers = { ...extra };
  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }
  return headers;
}

class ApiError extends Error {
  constructor(message, status, payload = null) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

async function apiFetch(path, options = {}) {
  const { method = "GET", body, headers = {}, skipAuthPrompt = false } = options;
  const reqHeaders = authHeaders(headers);
  const fetchOpts = { method, headers: reqHeaders };
  if (body !== undefined) {
    reqHeaders["Content-Type"] = "application/json";
    fetchOpts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, fetchOpts);
  if (resp.status === 401) {
    state.authRequired = true;
    updateAuthButton();
    if (!skipAuthPrompt) {
      openAuthModal("Authentication required. Enter your control-plane Bearer token.");
    }
    throw new ApiError("Unauthorized (401)", 401);
  }
  const contentType = resp.headers.get("content-type") || "";
  let data = null;
  if (contentType.includes("application/json")) {
    data = await resp.json().catch(() => null);
  } else {
    const txt = await resp.text().catch(() => "");
    data = { ok: resp.ok, text: txt };
  }
  if (!resp.ok || (data && data.ok === false)) {
    const msg =
      (data && (data.error || data.message)) ||
      (data && data.text) ||
      `HTTP ${resp.status}`;
    throw new ApiError(msg, resp.status, data);
  }
  return data;
}

function updateAuthButton() {
  const btn = $("btn-auth");
  if (!btn) return;
  btn.classList.toggle("authed", Boolean(state.token));
  btn.title = state.token
    ? "Bearer token saved in sessionStorage (click to change/clear)"
    : "Authentication / Bearer Token";
}

function openAuthModal(initialNotice = "") {
  if (state.authModalOpen) return;
  state.authModalOpen = true;
  openModal({
    title: "Control-Plane Authentication",
    bodyHtml: `
      ${initialNotice ? `<div class="form-error">${escapeHtml(initialNotice)}</div>` : ""}
      <div class="form-field">
        <label for="auth-token-input">Bearer Token (<code>SPEC_API_TOKEN</code>)</label>
        <input id="auth-token-input" class="input mono" type="password" placeholder="Paste Bearer token…" value="${escapeHtml(state.token)}" autocomplete="off" />
        <span class="form-hint">Stored only in browser <code>sessionStorage</code> for this tab; cleared automatically when the tab closes.</span>
      </div>
      <div id="auth-modal-err" class="form-error hidden"></div>
    `,
    footerHtml: `
      ${state.token ? `<button id="btn-auth-clear" class="btn btn-danger-outline btn-sm" type="button">Clear Token</button>` : ""}
      <button id="btn-auth-cancel" class="btn btn-ghost btn-sm" type="button">Cancel</button>
      <button id="btn-auth-save" class="btn btn-primary btn-sm" type="button">Verify &amp; Save</button>
    `,
    onMount({ body, footer }) {
      const input = body.querySelector("#auth-token-input");
      const errBox = body.querySelector("#auth-modal-err");
      const saveBtn = footer.querySelector("#btn-auth-save");
      input?.focus();

      const submit = async () => {
        const candidateToken = (input?.value || "").trim();
        errBox.classList.add("hidden");
        saveBtn.disabled = true;
        saveBtn.textContent = "Verifying…";
        const prev = state.token;
        state.token = candidateToken;
        try {
          await apiFetch("/api/spec/status", { skipAuthPrompt: true });
          if (candidateToken) {
            sessionStorage.setItem(TOKEN_KEY, candidateToken);
          } else {
            sessionStorage.removeItem(TOKEN_KEY);
          }
          state.authRequired = false;
          updateAuthButton();
          closeModal();
          toast("Authenticated successfully", "success");
          await refreshAll();
          if (state.drawerOpen) {
            connectAgentWs(true);
          }
        } catch (err) {
          state.token = prev;
          errBox.textContent =
            err.status === 401
              ? "Invalid Bearer token (401 Unauthorized)."
              : `Verification failed: ${err.message}`;
          errBox.classList.remove("hidden");
        } finally {
          saveBtn.disabled = false;
          saveBtn.textContent = "Verify & Save";
        }
      };

      saveBtn?.addEventListener("click", submit);
      input?.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          submit();
        }
      });
      footer.querySelector("#btn-auth-cancel")?.addEventListener("click", closeModal);
      footer.querySelector("#btn-auth-clear")?.addEventListener("click", () => {
        state.token = "";
        sessionStorage.removeItem(TOKEN_KEY);
        updateAuthButton();
        closeModal();
        toast("Token cleared from sessionStorage", "info");
        refreshAll();
      });
    },
    onClose() {
      state.authModalOpen = false;
    },
  });
}

/* ==========================================================================
   Data Fetching & Telemetry Rendering
   ========================================================================== */

async function fetchHealth() {
  try {
    const resp = await fetch("/health");
    if (resp.ok) {
      state.health = await resp.json();
    }
  } catch {
    state.health = null;
  }
}

function renderHealthBar() {
  const worker = state.status?.worker || null;
  const connected =
    worker != null
      ? Boolean(worker.connected)
      : Boolean(state.health?.worker_connected);

  const workerDot = $("worker-dot");
  const workerState = $("worker-state");
  const workerPill = $("worker-pill");
  if (workerDot && workerState) {
    workerDot.className = `status-dot ${connected ? "dot-ok" : "dot-danger"}`;
    workerState.textContent = connected ? "Connected" : "Offline";
    if (worker?.last_seen) {
      workerPill.title = `Worker ${connected ? "connected" : "offline"} · last heartbeat ${formatRelativeTime(worker.last_seen)}`;
    }
  }

  const metrics = worker?.metrics || {};
  const pressure = String(metrics.pressure || (connected ? "normal" : "unknown"));
  const used = Number(metrics.disk_used_bytes || 0);
  const limit = Number(metrics.disk_limit_bytes || 0);
  const freeSys = Number(metrics.system_free_bytes || 0);
  const pct = limit > 0 ? Math.min(100, Math.round((used / limit) * 100)) : 0;

  const gaugeFill = $("disk-gauge-fill");
  const badge = $("disk-pressure-badge");
  const usageText = $("disk-usage-text");
  const diskPill = $("disk-pill");

  if (gaugeFill) {
    gaugeFill.style.width = `${pct}%`;
    gaugeFill.className = `gauge-fill ${pressure === "blocked" ? "blocked" : pressure === "high" ? "high" : ""}`;
  }
  if (badge) {
    badge.textContent = pressure;
    badge.className = `badge ${
      pressure === "normal"
        ? "badge-success"
        : pressure === "high"
        ? "badge-warning"
        : pressure === "blocked"
        ? "badge-danger"
        : "badge-muted"
    }`;
  }
  if (usageText) {
    usageText.textContent = limit > 0 ? `${formatBytes(used)} / ${formatBytes(limit)}` : "";
  }
  if (diskPill && limit > 0) {
    diskPill.title = `Pool: ${formatBytes(used)} / ${formatBytes(limit)} (${pct}%) · Host free: ${formatBytes(freeSys)}`;
  }

  const wtCount = $("worktree-count");
  if (wtCount) {
    wtCount.textContent = String(metrics.worktrees ?? 0);
  }

  const tokEl = $("total-tokens");
  if (tokEl) {
    tokEl.textContent = formatTokens(state.status?.tokens_used ?? 0);
    tokEl.title = `${Number(state.status?.tokens_used ?? 0).toLocaleString()} total speculative tokens`;
  }
}

async function refreshAll({ silent = false } = {}) {
  const refreshIcon = document.querySelector("#btn-refresh .icon-spin-target");
  if (!silent && refreshIcon) refreshIcon.classList.add("spinning");

  await fetchHealth();
  try {
    const statusData = await apiFetch("/api/spec/status");
    state.status = statusData;
    state.projects = Array.isArray(statusData.projects) ? statusData.projects : [];
    if (statusData.agent_session_key && !state.agentSessionKey) {
      state.agentSessionKey = statusData.agent_session_key;
      const keyEl = $("drawer-session-key");
      if (keyEl) keyEl.textContent = state.agentSessionKey;
    }

    const params = new URLSearchParams({ limit: "100" });
    if (state.selectedProjectId) params.set("project_id", state.selectedProjectId);
    const candData = await apiFetch(`/api/spec/candidates?${params.toString()}`);
    state.candidates = Array.isArray(candData.candidates) ? candData.candidates : [];

    renderHealthBar();
    renderProjects();
    renderCandidatesList();

    if (state.selectedCandId) {
      const stillPresent = state.candidates.some((c) => c.cand_id === state.selectedCandId);
      if (stillPresent) {
        await loadCandidateDetail(state.selectedCandId, { preserveTab: true, silent: true });
      }
    }
  } catch (err) {
    renderHealthBar();
    if (err.status !== 401 && !silent) {
      toast(`Refresh failed: ${err.message}`, "error");
    }
  } finally {
    if (refreshIcon) refreshIcon.classList.remove("spinning");
  }
}

/* ==========================================================================
   Projects Panel
   ========================================================================== */

function projectStatusDotClass(status) {
  switch (status) {
    case "running":
      return "dot-info";
    case "idle":
      return "dot-ok";
    case "sleeping":
      return "dot-warn";
    case "error":
      return "dot-danger";
    default:
      return "dot-muted";
  }
}

function renderProjects() {
  const list = $("projects-list");
  const countEl = $("projects-count");
  if (!list) return;
  countEl.textContent = String(state.projects.length);

  if (!state.projects.length) {
    list.innerHTML = `
      <div class="empty-inline">
        No projects configured yet.<br/>Click <strong>+ New</strong> to register a repository.
      </div>`;
    return;
  }

  const allSelected = state.selectedProjectId === "";
  const cards = state.projects
    .map((p) => {
      const selected = p.project_id === state.selectedProjectId;
      const enabled = Boolean(p.enabled);
      const allowTests = Boolean(p.allow_test_edits);
      return `
        <div class="project-card ${selected ? "selected" : ""}" data-project-id="${escapeHtml(p.project_id)}">
          <div class="project-top">
            <div style="display:flex;align-items:center;gap:6px;min-width:0;">
              <span class="status-dot ${projectStatusDotClass(p.status)}" title="Status: ${escapeHtml(p.status)}"></span>
              <span class="project-name" title="${escapeHtml(p.name)}">${escapeHtml(p.name)}</span>
            </div>
            <span class="badge badge-muted mono">${escapeHtml(p.target_branch || "master")}</span>
          </div>

          <div class="project-meta">
            <span>${escapeHtml(p.status || "idle")}</span>
            <span>·</span>
            <span class="mono">${escapeHtml(formatTokens(p.speculative_tokens))} tok</span>
            ${p.last_head_sha ? `<span>·</span><span class="mono">${escapeHtml(shortSha(p.last_head_sha))}</span>` : ""}
            ${allowTests ? `<span class="badge badge-warning" title="Existing test edits opted-in">test-edits</span>` : ""}
          </div>

          <div class="project-actions">
            <label class="toggle-label" title="Toggle continuous autonomous mode">
              <input type="checkbox" class="proj-continuous-toggle" data-project-id="${escapeHtml(p.project_id)}" ${enabled ? "checked" : ""} />
              <span>Continuous</span>
            </label>
            <div class="project-action-btns">
              <button type="button" class="btn btn-ghost btn-xs proj-graveyard-btn" data-project-id="${escapeHtml(p.project_id)}" title="View Graveyard">Graveyard</button>
              <button type="button" class="btn btn-secondary btn-xs proj-run-btn" data-project-id="${escapeHtml(p.project_id)}" title="Trigger speculative batch now">Run Batch</button>
            </div>
          </div>
        </div>`;
    })
    .join("");

  list.innerHTML = `
    <button type="button" class="project-all-btn ${allSelected ? "selected" : ""}" data-select-all="1">
      <span>All Projects</span>
      <span class="count-pill">${state.projects.length}</span>
    </button>
    ${cards}`;
}

async function handleToggleContinuous(projectId, nextEnabled) {
  try {
    await apiFetch(`/api/spec/projects/${encodeURIComponent(projectId)}`, {
      method: "PATCH",
      body: { enabled: nextEnabled ? 1 : 0 },
    });
    toast(`Continuous mode ${nextEnabled ? "enabled" : "paused"}`, "success");
    await refreshAll({ silent: true });
  } catch (err) {
    toast(`Could not update project: ${err.message}`, "error");
    await refreshAll({ silent: true });
  }
}

async function handleRunBatch(projectId) {
  try {
    const res = await apiFetch(`/api/spec/projects/${encodeURIComponent(projectId)}/run`, {
      method: "POST",
      body: {},
    });
    toast(
      res.started ? "Speculative batch started" : "Batch is already running for this project",
      res.started ? "success" : "info",
    );
    await refreshAll({ silent: true });
  } catch (err) {
    toast(`Run batch failed: ${err.message}`, "error");
  }
}

async function openGraveyardModal(projectId) {
  const project = state.projects.find((p) => p.project_id === projectId);
  openModal({
    title: `Graveyard — ${project?.name || projectId}`,
    large: true,
    bodyHtml: `<div class="empty-inline">Loading rejected ideas…</div>`,
    footerHtml: `<button class="btn btn-secondary btn-sm" id="btn-grave-close" type="button">Close</button>`,
    onMount({ footer }) {
      footer.querySelector("#btn-grave-close")?.addEventListener("click", closeModal);
    },
  });

  try {
    const data = await apiFetch(
      `/api/spec/projects/${encodeURIComponent(projectId)}/graveyard?limit=100`,
    );
    const items = Array.isArray(data.graveyard) ? data.graveyard : [];
    const body = $("modal-body");
    if (!body) return;
    if (!items.length) {
      body.innerHTML = `<div class="empty-inline">Graveyard is empty — no rejected routes recorded yet.</div>`;
      return;
    }

    const renderItems = (query = "") => {
      const q = query.trim().toLowerCase();
      const filtered = q
        ? items.filter(
            (g) =>
              String(g.scope_hint || "").toLowerCase().includes(q) ||
              String(g.idea || "").toLowerCase().includes(q) ||
              String(g.reason || "").toLowerCase().includes(q) ||
              String(g.source || "").toLowerCase().includes(q),
          )
        : items;

      const listEl = body.querySelector("#grave-list");
      if (!listEl) return;
      if (!filtered.length) {
        listEl.innerHTML = `<div class="empty-inline">No matching graveyard entries.</div>`;
        return;
      }
      listEl.innerHTML = filtered
        .map(
          (g) => `
          <div class="grave-item">
            <div class="grave-head">
              <strong>${escapeHtml(g.scope_hint || "(untitled)")}</strong>
              <div style="display:flex;align-items:center;gap:6px;">
                <span class="badge ${g.source === "tournament_judge" ? "badge-purple" : g.source === "human_drop" ? "badge-danger" : "badge-muted"}">${escapeHtml(g.source || "unknown")}</span>
                <span class="mono-subtle">${escapeHtml(formatTimestamp(g.created_at))}</span>
              </div>
            </div>
            <div style="font-size:12.5px;color:var(--fg-subtle);">${escapeHtml(g.idea || "")}</div>
            ${g.reason ? `<div class="grave-reason">${escapeHtml(g.reason)}</div>` : ""}
          </div>`,
        )
        .join("");
    };

    body.innerHTML = `
      <input id="grave-search" class="input input-sm" type="search" placeholder="Filter ${items.length} graveyard entries by title, reason, or source…" />
      <div id="grave-list" style="display:flex;flex-direction:column;gap:8px;"></div>`;
    renderItems();
    body.querySelector("#grave-search")?.addEventListener("input", (e) => {
      renderItems(e.target.value);
    });
  } catch (err) {
    const body = $("modal-body");
    if (body) {
      body.innerHTML = `<div class="form-error">Failed to load graveyard: ${escapeHtml(err.message)}</div>`;
    }
  }
}

function openCreateProjectModal() {
  openModal({
    title: "Create Speculative Project",
    bodyHtml: `
      <div id="new-proj-err" class="form-error hidden"></div>
      <div class="form-field">
        <label for="np-name">Project Name *</label>
        <input id="np-name" class="input" type="text" placeholder="e.g. my-service" required />
      </div>
      <div class="form-field">
        <label for="np-repo">Public HTTPS Repository URL *</label>
        <input id="np-repo" class="input mono" type="url" placeholder="https://github.com/org/repo.git" required />
        <span class="form-hint">Must be credential-free HTTPS (no embedded tokens, query strings, or fragments).</span>
      </div>
      <div class="form-grid-2">
        <div class="form-field">
          <label for="np-branch">Target Branch</label>
          <input id="np-branch" class="input mono" type="text" value="main" />
        </div>
        <div class="form-field">
          <label for="np-routes">Routes per Batch (0–12)</label>
          <input id="np-routes" class="input mono" type="number" min="0" max="12" value="3" />
        </div>
      </div>
      <div class="form-field">
        <label for="np-gates">Deterministic Gate Commands (one per line)</label>
        <textarea id="np-gates" class="input mono" rows="3" placeholder="python -m pytest&#10;npm run build"></textarea>
      </div>
      <div class="form-grid-2">
        <div class="form-field">
          <label for="np-preview">Preview Command (optional)</label>
          <input id="np-preview" class="input mono" type="text" placeholder="npm run preview:capture" />
        </div>
        <div class="form-field">
          <label for="np-ttl">Candidate TTL (hours)</label>
          <input id="np-ttl" class="input mono" type="number" min="1" max="8760" value="72" />
        </div>
      </div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;padding-top:4px;">
        <label class="toggle-label">
          <input id="np-enabled" type="checkbox" />
          <span>Enable continuous polling immediately</span>
        </label>
        <label class="toggle-label">
          <input id="np-allow-tests" type="checkbox" />
          <span>Allow modifying existing tests (with regression guard)</span>
        </label>
      </div>
    `,
    footerHtml: `
      <button id="np-cancel" class="btn btn-ghost btn-sm" type="button">Cancel</button>
      <button id="np-submit" class="btn btn-primary btn-sm" type="button">Create Project</button>
    `,
    onMount({ body, footer }) {
      body.querySelector("#np-name")?.focus();
      footer.querySelector("#np-cancel")?.addEventListener("click", closeModal);
      const submitBtn = footer.querySelector("#np-submit");
      submitBtn?.addEventListener("click", async () => {
        const errEl = body.querySelector("#new-proj-err");
        errEl.classList.add("hidden");
        const gates = body
          .querySelector("#np-gates")
          .value.split("\n")
          .map((s) => s.trim())
          .filter(Boolean);
        const payload = {
          name: body.querySelector("#np-name").value.trim(),
          repo_source: body.querySelector("#np-repo").value.trim(),
          target_branch: body.querySelector("#np-branch").value.trim() || "main",
          routes_per_batch: Number(body.querySelector("#np-routes").value || 3),
          ttl_hours: Number(body.querySelector("#np-ttl").value || 72),
          gate_commands_json: gates,
          preview_command: body.querySelector("#np-preview").value.trim(),
          enabled: body.querySelector("#np-enabled").checked ? 1 : 0,
          allow_test_edits: body.querySelector("#np-allow-tests").checked ? 1 : 0,
        };
        submitBtn.disabled = true;
        submitBtn.textContent = "Creating…";
        try {
          const res = await apiFetch("/api/spec/projects", {
            method: "POST",
            body: payload,
          });
          closeModal();
          toast(`Project "${res.project?.name || payload.name}" created`, "success");
          if (res.project?.project_id) {
            state.selectedProjectId = res.project.project_id;
          }
          await refreshAll({ silent: true });
        } catch (err) {
          errEl.textContent = err.message;
          errEl.classList.remove("hidden");
        } finally {
          submitBtn.disabled = false;
          submitBtn.textContent = "Create Project";
        }
      });
    },
  });
}

/* ==========================================================================
   Candidates Review Queue
   ========================================================================== */

function getFilteredCandidates() {
  const q = state.searchQuery.trim().toLowerCase();
  return state.candidates.filter((c) => {
    if (state.statusFilter === "active") {
      if (!ACTIVE_STATUSES.has(c.status)) return false;
    } else if (state.statusFilter !== "all" && c.status !== state.statusFilter) {
      return false;
    }
    if (q) {
      const hay = `${c.title || ""} ${c.cand_id || ""} ${c.competition_group || ""} ${c.route_kind || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function renderCandidatesList() {
  const list = $("candidates-list");
  const countEl = $("candidates-count");
  if (!list) return;

  const filtered = getFilteredCandidates();
  countEl.textContent = String(filtered.length);

  if (!filtered.length) {
    list.innerHTML = `<div class="empty-inline">No candidates match the current filter.</div>`;
    return;
  }

  const projMap = new Map(state.projects.map((p) => [p.project_id, p.name]));

  list.innerHTML = filtered
    .map((c) => {
      const selected = c.cand_id === state.selectedCandId;
      const ttl = formatTtl(c.expires_at, c.status);
      const modTests = getModifiedTests(c);
      const projName = projMap.get(c.project_id) || c.project_id;
      return `
        <article class="cand-card ${selected ? "selected" : ""}" data-cand-id="${escapeHtml(c.cand_id)}">
          <div class="cand-card-top">
            <div class="cand-title">${escapeHtml(c.title || "(untitled candidate)")}</div>
            <span class="badge ${statusBadgeClass(c.status)}">${escapeHtml(c.status)}</span>
          </div>
          <div class="cand-badges">
            <span class="badge badge-muted">${escapeHtml(c.route_kind || "optimization")}</span>
            ${c.competition_group ? `<span class="badge badge-purple" title="Competition group">⚔ ${escapeHtml(c.competition_group)}</span>` : ""}
            <span class="badge badge-muted" title="Review rounds">R${escapeHtml(c.review_rounds ?? 0)}</span>
            ${ttl ? `<span class="badge ${ttl.cls}">${escapeHtml(ttl.text)}</span>` : ""}
            ${
              modTests.length
                ? `<span class="badge badge-warning" title="Existing test files modified: ${escapeHtml(modTests.join(", "))}">⚠ ${modTests.length} test${modTests.length === 1 ? "" : "s"} modified</span>`
                : ""
            }
          </div>
          <div class="cand-foot">
            <span>${escapeHtml(projName)}</span>
            <span class="mono">${escapeHtml(formatTokens(c.tokens_used))} tok · ${escapeHtml(formatRelativeTime(c.updated_at || c.created_at))}</span>
          </div>
        </article>`;
    })
    .join("");
}

/* ==========================================================================
   Candidate Inspector
   ========================================================================== */

function revokePreviewBlobs() {
  for (const url of state.previewBlobUrls) {
    try {
      URL.revokeObjectURL(url);
    } catch {
      // ignore
    }
  }
  state.previewBlobUrls = [];
}

async function loadCandidateDetail(candId, { preserveTab = false, silent = false } = {}) {
  state.selectedCandId = candId;
  renderCandidatesList();

  const seq = ++state.inspectorSeq;
  if (!preserveTab) {
    state.activeTab = "report";
    state.activePreviewPath = "";
  }

  $("inspector-empty").classList.add("hidden");
  $("inspector-content").classList.remove("hidden");

  try {
    const [detailData, eventsData] = await Promise.all([
      apiFetch(`/api/spec/candidates/${encodeURIComponent(candId)}`),
      apiFetch(`/api/spec/candidates/${encodeURIComponent(candId)}/events?limit=150`).catch(
        () => ({ ok: true, events: [] }),
      ),
    ]);
    if (seq !== state.inspectorSeq) return;

    state.candidateDetail = detailData.candidate;
    state.candidateEvents = Array.isArray(eventsData.events) ? eventsData.events : [];
    renderInspector();
  } catch (err) {
    if (seq !== state.inspectorSeq) return;
    if (!silent && err.status !== 401) {
      toast(`Failed to load candidate: ${err.message}`, "error");
    }
  }
}

function renderInspector() {
  const c = state.candidateDetail;
  if (!c) {
    $("inspector-empty").classList.remove("hidden");
    $("inspector-content").classList.add("hidden");
    return;
  }

  const proj = state.projects.find((p) => p.project_id === c.project_id);
  const ttl = formatTtl(c.expires_at, c.status);
  const modTests = getModifiedTests(c);
  const actionable = c.status === "staged" || c.status === "kept";

  $("inspector-badges").innerHTML = `
    <span class="badge ${statusBadgeClass(c.status)}">${escapeHtml(c.status)}</span>
    <span class="badge badge-muted">${escapeHtml(c.route_kind || "optimization")}</span>
    ${c.competition_group ? `<span class="badge badge-purple">⚔ ${escapeHtml(c.competition_group)}</span>` : ""}
    <span class="badge badge-muted">Rounds: ${escapeHtml(c.review_rounds ?? 0)}</span>
    ${ttl ? `<span class="badge ${ttl.cls}">${escapeHtml(ttl.text)}</span>` : ""}
  `;

  $("inspector-title").textContent = c.title || "(untitled candidate)";
  $("inspector-meta").innerHTML = `
    <span><strong>${escapeHtml(proj?.name || c.project_id)}</strong></span>
    <span class="mono" title="Candidate ID">${escapeHtml(c.cand_id)}</span>
    <span class="mono" title="Base -> Head SHA">${escapeHtml(shortSha(c.base_sha))} → ${escapeHtml(shortSha(c.head_sha))}</span>
    ${c.branch_ref ? `<span class="mono">${escapeHtml(c.branch_ref)}</span>` : ""}
    <span class="mono">${escapeHtml(formatTokens(c.tokens_used))} tok</span>
  `;

  // Action button states (all human actions require staged or kept per perform_action)
  $("act-keep").disabled = !(c.status === "staged");
  $("act-steer").disabled = !actionable;
  $("act-drop").disabled = !actionable;
  $("act-export").disabled = !actionable;
  $("act-merge").disabled = !(actionable && Boolean(c.head_sha));

  // Modified existing tests warning banner
  const warnBanner = $("test-warning-banner");
  if (modTests.length) {
    const reg = c.preview_meta_json?.baseline_test_regression;
    const regNote =
      reg && reg.checked
        ? reg.passed
          ? " · Baseline regression check PASSED."
          : " · WARNING: Baseline regression check FAILED!"
        : "";
    warnBanner.innerHTML = `
      <div><strong>⚠ Existing test files modified (${modTests.length})</strong> — audit test changes carefully before merging.${escapeHtml(regNote)}</div>
      <div class="test-warning-files">${modTests.map((f) => `<span class="test-warning-file">${escapeHtml(f)}</span>`).join("")}</div>`;
    warnBanner.classList.remove("hidden");
  } else {
    warnBanner.classList.add("hidden");
    warnBanner.innerHTML = "";
  }

  // Tab 1: Human Report + Route Plan + Steer Notes
  const steerNotes = Array.isArray(c.steer_notes_json) ? c.steer_notes_json : [];
  $("panel-report").innerHTML = `
    <div class="report-layout">
      <div class="md-body">${renderMarkdown(c.human_report_md)}</div>
      ${
        c.route_plan
          ? `<details class="info-box">
               <summary>Original Route Plan</summary>
               <div class="info-box-content md-body">${renderMarkdown(c.route_plan)}</div>
             </details>`
          : ""
      }
      ${
        steerNotes.length
          ? `<details class="info-box" open>
               <summary>Human Steer Notes (${steerNotes.length})</summary>
               <div class="info-box-content" style="display:flex;flex-direction:column;gap:6px;">
                 ${steerNotes
                   .map(
                     (n) => `
                     <div style="padding:6px 8px;border:1px solid var(--border);border-radius:6px;background:var(--surface);">
                       <div class="mono-subtle">${escapeHtml(n.actor || "human")} · ${escapeHtml(formatTimestamp(n.at))}</div>
                       <div style="margin-top:3px;white-space:pre-wrap;">${escapeHtml(n.notes || "")}</div>
                     </div>`,
                   )
                   .join("")}
               </div>
             </details>`
          : ""
      }
    </div>`;

  // Tab 2: Diff Viewer
  const diffRes = renderDiffViewer(c.diff_text, modTests);
  $("panel-diff").innerHTML = diffRes.html;
  const diffBadge = $("tab-diff-count");
  if (diffRes.fileCount > 0) {
    diffBadge.textContent = String(diffRes.fileCount);
    diffBadge.classList.remove("hidden");
  } else {
    diffBadge.classList.add("hidden");
  }

  // Bind Diff file jump & copy button
  $("panel-diff").querySelectorAll(".diff-file-jump").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.getElementById(btn.dataset.jumpTarget);
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
  $("btn-copy-raw-diff")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(c.diff_text || "");
      toast("Unified diff copied to clipboard", "success");
    } catch {
      toast("Clipboard copy failed", "error");
    }
  });

  // Tab 3: Gates & Evidence
  $("panel-evidence").innerHTML = renderEvidencePanel(c, state.candidateEvents);
  const evBadge = $("tab-evidence-count");
  if (state.candidateEvents.length > 0) {
    evBadge.textContent = String(state.candidateEvents.length);
    evBadge.classList.remove("hidden");
  } else {
    evBadge.classList.add("hidden");
  }

  // Tab 4: Preview Artifacts
  renderPreviewTab(c);

  setActiveTab(state.activeTab);
}

function setActiveTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".inspector-tabs .tab-btn").forEach((btn) => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".inspector-body .tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${tabName}`);
  });
}

function renderPreviewTab(candidate) {
  revokePreviewBlobs();
  const panel = $("panel-preview");
  const meta = candidate?.preview_meta_json || {};
  const artifacts = Array.isArray(meta.artifacts) ? meta.artifacts : [];
  const previewMeta = meta.preview || null;

  const badge = $("tab-preview-count");
  if (artifacts.length > 0) {
    badge.textContent = String(artifacts.length);
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }

  if (!artifacts.length && !previewMeta) {
    panel.innerHTML = `<div class="empty-inline">No preview artifacts captured for this candidate.</div>`;
    return;
  }

  const previewRunHtml =
    previewMeta && typeof previewMeta === "object"
      ? `<div class="gate-card">
           <div class="gate-head">
             <span class="gate-cmd">${escapeHtml(previewMeta.command || "preview_command")}</span>
             <span class="badge ${previewMeta.code === 0 ? "badge-success" : "badge-danger"}">exit ${escapeHtml(previewMeta.code ?? "?")}</span>
           </div>
           ${previewMeta.output ? `<pre class="gate-output">${escapeHtml(previewMeta.output)}</pre>` : ""}
         </div>`
      : "";

  if (!artifacts.length) {
    panel.innerHTML = `<div class="preview-layout">${previewRunHtml}<div class="empty-inline">Preview command ran with no saved artifact files.</div></div>`;
    return;
  }

  const selectedPath =
    artifacts.some((a) => a.path === state.activePreviewPath)
      ? state.activePreviewPath
      : artifacts[0].path;
  state.activePreviewPath = selectedPath;

  panel.innerHTML = `
    <div class="preview-layout">
      ${previewRunHtml}
      <div class="preview-tabs" id="preview-artifact-tabs">
        ${artifacts
          .map(
            (a) => `
            <button type="button" class="filter-pill ${a.path === selectedPath ? "active" : ""}" data-preview-path="${escapeHtml(a.path)}">
              ${escapeHtml(a.path)}
            </button>`,
          )
          .join("")}
      </div>
      <div class="preview-frame-box" id="preview-stage">
        <div class="empty-inline">Loading preview artifact…</div>
      </div>
    </div>`;

  panel.querySelectorAll("[data-preview-path]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activePreviewPath = btn.dataset.previewPath;
      panel.querySelectorAll("[data-preview-path]").forEach((b) => {
        b.classList.toggle("active", b.dataset.previewPath === state.activePreviewPath);
      });
      loadPreviewArtifact(candidate.cand_id, state.activePreviewPath);
    });
  });

  loadPreviewArtifact(candidate.cand_id, selectedPath);
}

async function loadPreviewArtifact(candId, artifactPath) {
  const stage = $("preview-stage");
  if (!stage) return;
  const seq = state.inspectorSeq;
  stage.innerHTML = `<div class="empty-inline">Fetching ${escapeHtml(artifactPath)}…</div>`;

  const encodedPath = artifactPath
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
  const url = `/api/spec/candidates/${encodeURIComponent(candId)}/preview/${encodedPath}`;

  try {
    const resp = await fetch(url, { headers: authHeaders() });
    if (resp.status === 401) {
      openAuthModal("Authentication required to load preview artifacts.");
      return;
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    if (seq !== state.inspectorSeq) return;

    const ctype = (resp.headers.get("content-type") || "").toLowerCase();
    const ext = artifactPath.split(".").pop()?.toLowerCase() || "";
    const isImage =
      ctype.startsWith("image/") ||
      ["png", "jpg", "jpeg", "webp", "gif", "bmp", "svg"].includes(ext);
    const isHtml = ctype.includes("text/html") || ext === "html" || ext === "htm";

    const header = `
      <div class="preview-frame-head">
        <span>${escapeHtml(artifactPath)}</span>
        <span class="badge badge-muted">${escapeHtml(ctype || ext || "artifact")}</span>
      </div>`;

    if (isImage) {
      const blob = await resp.blob();
      const blobUrl = URL.createObjectURL(blob);
      state.previewBlobUrls.push(blobUrl);
      stage.innerHTML = `
        ${header}
        <div class="preview-img-wrap">
          <img class="preview-img" src="${blobUrl}" alt="${escapeHtml(artifactPath)}" />
        </div>`;
    } else if (isHtml) {
      const htmlText = await resp.text();
      stage.innerHTML = header;
      const iframe = document.createElement("iframe");
      iframe.className = "preview-iframe";
      iframe.setAttribute("sandbox", "allow-scripts");
      iframe.setAttribute("referrerpolicy", "no-referrer");
      iframe.title = `Sandboxed preview: ${artifactPath}`;
      iframe.srcdoc = htmlText;
      stage.appendChild(iframe);
    } else {
      const text = await resp.text();
      stage.innerHTML = `
        ${header}
        <pre class="gate-output" style="max-height:520px;">${escapeHtml(text)}</pre>`;
    }
  } catch (err) {
    if (seq !== state.inspectorSeq) return;
    stage.innerHTML = `<div class="form-error" style="margin:12px;">Failed to load preview: ${escapeHtml(err.message)}</div>`;
  }
}

/* ==========================================================================
   Human-Only Actions (Keep, Steer, Drop, Export, Merge)
   ========================================================================== */

async function executeCandidateAction(candId, action, payload) {
  const body = {
    action_id: generateActionId(action),
    ...payload,
  };
  return apiFetch(`/api/spec/candidates/${encodeURIComponent(candId)}/${action}`, {
    method: "POST",
    body,
  });
}

async function handleKeepAction() {
  const c = state.candidateDetail;
  if (!c) return;
  const btn = $("act-keep");
  btn.disabled = true;
  try {
    await executeCandidateAction(c.cand_id, "keep", {});
    toast("Candidate kept — TTL frozen", "success");
    await refreshAll({ silent: true });
  } catch (err) {
    toast(`Keep failed: ${err.message}`, "error");
    btn.disabled = false;
  }
}

function openSteerModal() {
  const c = state.candidateDetail;
  if (!c) return;
  openModal({
    title: `Steer Candidate — ${c.title}`,
    bodyHtml: `
      <div id="steer-err" class="form-error hidden"></div>
      <div class="form-field">
        <label for="steer-notes">Revision Guidance *</label>
        <textarea id="steer-notes" class="input" rows="5" placeholder="Describe the concrete adjustments you want the Coder to apply on the next revision round…"></textarea>
        <span class="form-hint">Rehydrates the candidate from <code>${escapeHtml(c.branch_ref || shortSha(c.head_sha))}</code> and restarts deterministic gates + independent review.</span>
      </div>
    `,
    footerHtml: `
      <button id="steer-cancel" class="btn btn-ghost btn-sm" type="button">Cancel</button>
      <button id="steer-submit" class="btn btn-primary btn-sm" type="button">Send Steer Notes</button>
    `,
    onMount({ body, footer }) {
      body.querySelector("#steer-notes")?.focus();
      footer.querySelector("#steer-cancel")?.addEventListener("click", closeModal);
      const submitBtn = footer.querySelector("#steer-submit");
      submitBtn?.addEventListener("click", async () => {
        const notes = body.querySelector("#steer-notes").value.trim();
        const errEl = body.querySelector("#steer-err");
        if (!notes) {
          errEl.textContent = "Please enter revision guidance.";
          errEl.classList.remove("hidden");
          return;
        }
        submitBtn.disabled = true;
        submitBtn.textContent = "Submitting…";
        try {
          await executeCandidateAction(c.cand_id, "steer", { notes });
          closeModal();
          toast("Candidate steered — revising started", "success");
          await refreshAll({ silent: true });
        } catch (err) {
          errEl.textContent = err.message;
          errEl.classList.remove("hidden");
          submitBtn.disabled = false;
          submitBtn.textContent = "Send Steer Notes";
        }
      });
    },
  });
}

function openDropModal() {
  const c = state.candidateDetail;
  if (!c) return;
  openModal({
    title: `Drop Candidate — ${c.title}`,
    bodyHtml: `
      <div id="drop-err" class="form-error hidden"></div>
      <div class="danger-box">
        Dropping deletes the candidate's ref &amp; artifacts and records this idea in the project's <strong>Graveyard</strong> so Recon will not propose it again.
      </div>
      <div class="form-field">
        <label for="drop-reason">Reason (recorded in Graveyard)</label>
        <textarea id="drop-reason" class="input" rows="3" placeholder="e.g. Over-engineered for current throughput, or conflicts with planned refactor…"></textarea>
      </div>
    `,
    footerHtml: `
      <button id="drop-cancel" class="btn btn-ghost btn-sm" type="button">Cancel</button>
      <button id="drop-submit" class="btn btn-danger btn-sm" type="button">Confirm Drop</button>
    `,
    onMount({ body, footer }) {
      body.querySelector("#drop-reason")?.focus();
      footer.querySelector("#drop-cancel")?.addEventListener("click", closeModal);
      const submitBtn = footer.querySelector("#drop-submit");
      submitBtn?.addEventListener("click", async () => {
        const reason = body.querySelector("#drop-reason").value.trim() || "human drop";
        const errEl = body.querySelector("#drop-err");
        submitBtn.disabled = true;
        submitBtn.textContent = "Dropping…";
        try {
          await executeCandidateAction(c.cand_id, "drop", { reason });
          closeModal();
          state.selectedCandId = "";
          state.candidateDetail = null;
          renderInspector();
          toast("Candidate dropped and added to graveyard", "info");
          await refreshAll({ silent: true });
        } catch (err) {
          errEl.textContent = err.message;
          errEl.classList.remove("hidden");
          submitBtn.disabled = false;
          submitBtn.textContent = "Confirm Drop";
        }
      });
    },
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function base64ToUint8Array(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

async function handleExportAction() {
  const c = state.candidateDetail;
  if (!c) return;
  const btn = $("act-export");
  btn.disabled = true;
  try {
    const res = await executeCandidateAction(c.cand_id, "export", {
      include_bundle: true,
    });
    const patchText = String(res.patch || "");
    const bundleB64 = String(res.bundle_base64 || "");
    const bundleFilename = String(res.bundle_filename || `${c.cand_id}.bundle`);
    const patchFilename = `${c.cand_id}.patch`;
    const instr = res.apply_instructions;
    let instrText = "";
    if (typeof instr === "string") {
      instrText = instr;
    } else if (instr && typeof instr === "object") {
      const parts = [];
      if (instr.patch) parts.push(`# Apply unified patch\n${instr.patch}`);
      if (instr.bundle) parts.push(`# Fetch & apply git bundle\n${instr.bundle}`);
      instrText = parts.join("\n\n");
    }

    openModal({
      title: `Export — ${c.title}`,
      large: true,
      bodyHtml: `
        <div class="inspector-meta" style="margin-top:0;">
          <span class="mono">Head: ${escapeHtml(shortSha(res.head_sha || c.head_sha))}</span>
          <span class="mono">Base: ${escapeHtml(shortSha(res.base_sha || c.base_sha))}</span>
          ${res.branch_ref ? `<span class="mono">Ref: ${escapeHtml(res.branch_ref)}</span>` : ""}
        </div>
        ${res.bundle_error ? `<div class="form-error">Bundle skipped: ${escapeHtml(res.bundle_error)}</div>` : ""}
        ${
          instrText
            ? `<div class="form-field">
                 <label>Local Apply Instructions</label>
                 <div class="code-block-copy"><pre>${escapeHtml(instrText)}</pre></div>
               </div>`
            : ""
        }
        <div class="form-field">
          <label>Unified Patch (${patchText.length.toLocaleString()} chars)</label>
          <div class="code-block-copy"><pre>${escapeHtml(patchText || "(empty patch)")}</pre></div>
        </div>
      `,
      footerHtml: `
        <button id="exp-copy-patch" class="btn btn-secondary btn-sm" type="button">Copy Patch</button>
        <button id="exp-dl-patch" class="btn btn-secondary btn-sm" type="button">Download .patch</button>
        ${bundleB64 ? `<button id="exp-dl-bundle" class="btn btn-primary btn-sm" type="button">Download .bundle</button>` : ""}
        <button id="exp-close" class="btn btn-ghost btn-sm" type="button">Close</button>
      `,
      onMount({ footer }) {
        footer.querySelector("#exp-close")?.addEventListener("click", closeModal);
        footer.querySelector("#exp-copy-patch")?.addEventListener("click", async () => {
          try {
            await navigator.clipboard.writeText(patchText);
            toast("Patch copied to clipboard", "success");
          } catch {
            toast("Clipboard copy failed", "error");
          }
        });
        footer.querySelector("#exp-dl-patch")?.addEventListener("click", () => {
          downloadBlob(new Blob([patchText], { type: "text/x-diff;charset=utf-8" }), patchFilename);
        });
        footer.querySelector("#exp-dl-bundle")?.addEventListener("click", () => {
          try {
            const bytes = base64ToUint8Array(bundleB64);
            downloadBlob(new Blob([bytes], { type: "application/octet-stream" }), bundleFilename);
          } catch (err) {
            toast(`Bundle decode failed: ${err.message}`, "error");
          }
        });
      },
    });
  } catch (err) {
    toast(`Export failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

function openMergeModal() {
  const c = state.candidateDetail;
  if (!c) return;
  const proj = state.projects.find((p) => p.project_id === c.project_id);
  const targetBranch = proj?.target_branch || "master";
  const defaultTargetSha = proj?.last_head_sha || c.base_sha || "";
  const modTests = getModifiedTests(c);

  openModal({
    title: "Confirm Local CAS Merge",
    bodyHtml: `
      <div id="merge-err" class="form-error hidden"></div>
      <div class="danger-box">
        <strong>Human-only CAS Merge</strong>: Merges candidate <code>${escapeHtml(shortSha(c.head_sha))}</code> locally into <code>${escapeHtml(targetBranch)}</code> inside the sandbox repository. This operation is SHA-pinned (Compare-And-Swap) and <strong>never auto-pushes</strong> to any remote.
      </div>
      ${
        modTests.length
          ? `<div class="form-error">⚠ This candidate modified ${modTests.length} existing test file(s): ${escapeHtml(modTests.join(", "))}</div>`
          : ""
      }
      <div class="form-field">
        <label>Target Branch</label>
        <input class="input mono" type="text" value="${escapeHtml(targetBranch)}" readonly />
      </div>
      <div class="form-field">
        <label>Approved Candidate <code>head_sha</code></label>
        <input id="merge-head-sha" class="input mono" type="text" value="${escapeHtml(c.head_sha || "")}" readonly />
      </div>
      <div class="form-field">
        <label for="merge-target-sha">Approved Target Branch SHA (<code>approved_target_sha</code>) *</label>
        <input id="merge-target-sha" class="input mono" type="text" value="${escapeHtml(defaultTargetSha)}" placeholder="40-char git SHA of target branch" />
        <span class="form-hint">Must match the current tip of <code>${escapeHtml(targetBranch)}</code> (prevents race conditions).</span>
      </div>
      <label class="confirm-check">
        <input id="merge-confirm-cb" type="checkbox" />
        <span>I have reviewed the human report, diff, and gate evidence, verified both SHAs above, and explicitly authorize this local CAS merge.</span>
      </label>
    `,
    footerHtml: `
      <button id="merge-cancel" class="btn btn-ghost btn-sm" type="button">Cancel</button>
      <button id="merge-submit" class="btn btn-danger btn-sm" type="button" disabled>Execute Local Merge</button>
    `,
    onMount({ body, footer }) {
      const cb = body.querySelector("#merge-confirm-cb");
      const targetInput = body.querySelector("#merge-target-sha");
      const headInput = body.querySelector("#merge-head-sha");
      const submitBtn = footer.querySelector("#merge-submit");
      const errEl = body.querySelector("#merge-err");

      const updateEnabled = () => {
        submitBtn.disabled = !(
          cb.checked &&
          headInput.value.trim().length > 0 &&
          targetInput.value.trim().length > 0
        );
      };
      cb.addEventListener("change", updateEnabled);
      targetInput.addEventListener("input", updateEnabled);
      footer.querySelector("#merge-cancel")?.addEventListener("click", closeModal);

      submitBtn.addEventListener("click", async () => {
        errEl.classList.add("hidden");
        submitBtn.disabled = true;
        submitBtn.textContent = "Merging…";
        try {
          await executeCandidateAction(c.cand_id, "merge", {
            confirm: true,
            approved_head_sha: headInput.value.trim(),
            approved_target_sha: targetInput.value.trim(),
            target_branch: targetBranch,
          });
          closeModal();
          toast("Candidate merged locally via CAS!", "success");
          await refreshAll({ silent: true });
        } catch (err) {
          errEl.textContent = err.message;
          errEl.classList.remove("hidden");
          submitBtn.textContent = "Execute Local Merge";
          updateEnabled();
        }
      });
    },
  });
}

/* ==========================================================================
   SpecAgent Chat Drawer (WS via ws-ticket + HTTP POST fallback)
   ========================================================================== */

function setAgentWsIndicator(connected, label) {
  state.agentWsConnected = connected;
  const dotCls = `status-dot ${connected ? "dot-ok" : "dot-muted"}`;
  const topDot = $("agent-ws-dot");
  const drawerDot = $("drawer-ws-dot");
  const drawerLabel = $("drawer-ws-label");
  if (topDot) topDot.className = dotCls;
  if (drawerDot) drawerDot.className = dotCls;
  if (drawerLabel) drawerLabel.textContent = label || (connected ? "WS Live" : "HTTP Fallback");
}

function setAgentRunning(running) {
  state.agentRunning = running;
  $("btn-agent-stop")?.classList.toggle("hidden", !running);
  const sendBtn = $("btn-agent-send");
  if (sendBtn) sendBtn.disabled = running;
}

function openAgentDrawer() {
  state.drawerOpen = true;
  $("agent-backdrop").classList.remove("hidden");
  $("agent-drawer").classList.remove("hidden");
  loadAgentHistory();
  if (!state.agentWs || state.agentWs.readyState > WebSocket.OPEN) {
    connectAgentWs();
  }
  $("agent-input")?.focus();
}

function closeAgentDrawer() {
  state.drawerOpen = false;
  $("agent-backdrop").classList.add("hidden");
  $("agent-drawer").classList.add("hidden");
}

async function loadAgentHistory() {
  const skey = state.agentSessionKey || state.status?.agent_session_key;
  if (!skey) return;
  $("drawer-session-key").textContent = skey;
  try {
    const data = await apiFetch(
      `/api/spec/agent/sessions/${encodeURIComponent(skey)}/turns?limit=50`,
    );
    state.agentTurns = Array.isArray(data.turns) ? data.turns : [];
    renderAgentMessages();
  } catch {
    // non-fatal
  }
}

async function connectAgentWs(force = false) {
  if (force && state.agentWs) {
    try {
      state.agentWs.close();
    } catch {
      // ignore
    }
    state.agentWs = null;
  }
  if (state.agentWs && state.agentWs.readyState <= WebSocket.OPEN) return;

  setAgentWsIndicator(false, "Connecting…");
  let ticketQuery = "";
  if (state.token) {
    try {
      const ticketRes = await apiFetch("/api/spec/agent/ws-ticket", {
        method: "POST",
        body: {},
        skipAuthPrompt: true,
      });
      if (ticketRes?.ticket) {
        ticketQuery = `?ticket=${encodeURIComponent(ticketRes.ticket)}`;
      }
    } catch {
      setAgentWsIndicator(false, "HTTP Fallback");
      return;
    }
  }

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/api/spec/agent/ws${ticketQuery}`;
  try {
    const ws = new WebSocket(wsUrl);
    state.agentWs = ws;

    ws.addEventListener("open", () => {
      setAgentWsIndicator(true, "WS Live");
    });

    ws.addEventListener("close", () => {
      if (state.agentWs === ws) {
        state.agentWs = null;
        setAgentWsIndicator(false, "HTTP Fallback");
      }
    });

    ws.addEventListener("error", () => {
      setAgentWsIndicator(false, "HTTP Fallback");
    });

    ws.addEventListener("message", (event) => {
      let msg = null;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      handleAgentWsEvent(msg);
    });
  } catch {
    setAgentWsIndicator(false, "HTTP Fallback");
  }
}

function handleAgentWsEvent(msg) {
  const kind = msg.type || msg.event_type || "";
  const payload = msg.payload || {};

  if (kind === "connected") {
    if (msg.session_key) {
      state.agentSessionKey = msg.session_key;
      $("drawer-session-key").textContent = msg.session_key;
      loadAgentHistory();
    }
    return;
  }

  if (kind === "turn_started") {
    setAgentRunning(true);
    state.agentLive = {
      user_content: payload.user_text || state.agentLive?.user_content || "",
      assistant_content: "",
      assistant_reasoning: "",
      tool_calls: [],
    };
    renderAgentMessages();
    return;
  }

  if (kind === "reasoning_delta") {
    if (!state.agentLive) {
      state.agentLive = { user_content: "", assistant_content: "", assistant_reasoning: "", tool_calls: [] };
    }
    state.agentLive.assistant_reasoning += String(payload.delta || payload.text || msg.delta || "");
    renderAgentMessages();
    return;
  }

  if (
    kind === "text_delta" ||
    kind === "assistant_delta" ||
    kind === "content_delta" ||
    kind === "delta"
  ) {
    if (!state.agentLive) {
      state.agentLive = { user_content: "", assistant_content: "", assistant_reasoning: "", tool_calls: [] };
    }
    state.agentLive.assistant_content += String(payload.delta || payload.text || msg.delta || "");
    renderAgentMessages();
    return;
  }

  if (kind === "tool_call_started" || kind === "tool_call_completed") {
    if (!state.agentLive) {
      state.agentLive = { user_content: "", assistant_content: "", assistant_reasoning: "", tool_calls: [] };
    }
    const name = payload.tool_name || payload.name || "tool";
    const id = payload.tool_call_id || payload.id || `${name}_${state.agentLive.tool_calls.length}`;
    const existing = state.agentLive.tool_calls.find((t) => t.id === id);
    if (existing) {
      existing.done = kind === "tool_call_completed";
      if (payload.result_preview) existing.preview = payload.result_preview;
    } else {
      state.agentLive.tool_calls.push({
        id,
        name,
        done: kind === "tool_call_completed",
        preview: payload.result_preview || "",
      });
    }
    renderAgentMessages();
    return;
  }

  if (kind === "turn_complete") {
    setAgentRunning(false);
    state.agentLive = null;
    loadAgentHistory();
    refreshAll({ silent: true });
    return;
  }

  if (kind === "stopped") {
    setAgentRunning(false);
    state.agentLive = null;
    loadAgentHistory();
    toast("SpecAgent turn stopped", "info");
    return;
  }

  if (kind === "error") {
    setAgentRunning(false);
    toast(`SpecAgent error: ${msg.message || payload.error || msg.code || "unknown"}`, "error");
    state.agentLive = null;
    loadAgentHistory();
  }
}

function renderAgentMessages() {
  const box = $("agent-messages");
  if (!box) return;

  const items = [...state.agentTurns];
  if (state.agentLive) {
    items.push({ ...state.agentLive, _live: true });
  }

  if (!items.length) {
    box.innerHTML = `<div class="empty-inline">Ask SpecAgent to explore a repository, explain a candidate, or schedule batches.</div>`;
    return;
  }

  box.innerHTML = items
    .map((t) => {
      const tools = Array.isArray(t.tool_calls) ? t.tool_calls : [];
      const toolHtml = tools.length
        ? `<div class="chat-tools">${tools
            .map(
              (tc) =>
                `<span class="tool-chip" title="${escapeHtml(tc.preview || "")}">${tc.done === false ? "⋯ " : "✓ "}${escapeHtml(tc.name || tc.tool_name || "tool")}</span>`,
            )
            .join("")}</div>`
        : "";

      const reasoningHtml = t.assistant_reasoning
        ? `<details class="reasoning-box"><summary>Reasoning</summary><pre>${escapeHtml(t.assistant_reasoning)}</pre></details>`
        : "";

      const replyContent = t.assistant_content || (t._live ? "Thinking…" : "");

      return `
        ${
          t.user_content
            ? `<div class="chat-msg chat-user"><div class="chat-bubble">${escapeHtml(t.user_content)}</div></div>`
            : ""
        }
        <div class="chat-msg chat-assistant">
          ${toolHtml}
          ${reasoningHtml}
          <div class="chat-bubble md-body">${renderMarkdown(replyContent)}</div>
        </div>`;
    })
    .join("");

  box.scrollTop = box.scrollHeight;
}

async function sendAgentMessage(text) {
  const clean = text.trim();
  if (!clean || state.agentRunning) return;

  const skey = state.agentSessionKey || state.status?.agent_session_key || "";
  if (state.agentWs && state.agentWs.readyState === WebSocket.OPEN) {
    setAgentRunning(true);
    state.agentLive = {
      user_content: clean,
      assistant_content: "",
      assistant_reasoning: "",
      tool_calls: [],
    };
    renderAgentMessages();
    state.agentWs.send(
      JSON.stringify({
        type: "chat",
        session_key: skey || undefined,
        text: clean,
      }),
    );
    return;
  }

  // Fallback to POST /api/spec/agent/turn
  setAgentRunning(true);
  state.agentLive = {
    user_content: clean,
    assistant_content: "Running turn via HTTP…",
    assistant_reasoning: "",
    tool_calls: [],
  };
  renderAgentMessages();
  try {
    const res = await apiFetch("/api/spec/agent/turn", {
      method: "POST",
      body: {
        session_key: skey || undefined,
        text: clean,
      },
    });
    state.agentLive = null;
    if (res.session_key && !state.agentSessionKey) {
      state.agentSessionKey = res.session_key;
    }
    await loadAgentHistory();
    await refreshAll({ silent: true });
  } catch (err) {
    state.agentLive = null;
    renderAgentMessages();
    toast(`SpecAgent turn failed: ${err.message}`, "error");
  } finally {
    setAgentRunning(false);
  }
}

/* ==========================================================================
   Event Wiring & Bootstrap
   ========================================================================== */

function bindEvents() {
  $("btn-refresh")?.addEventListener("click", () => refreshAll());
  $("btn-theme")?.addEventListener("click", cycleTheme);
  $("btn-auth")?.addEventListener("click", () => openAuthModal());
  $("btn-new-project")?.addEventListener("click", openCreateProjectModal);

  $("btn-toggle-sidebar")?.addEventListener("click", () => {
    $("projects-pane")?.classList.toggle("mobile-open");
  });

  // Modal close handlers
  $("modal-close")?.addEventListener("click", closeModal);
  $("modal-backdrop")?.addEventListener("click", (e) => {
    if (e.target === $("modal-backdrop")) closeModal();
  });

  // Projects list delegation
  $("projects-list")?.addEventListener("click", (e) => {
    const allBtn = e.target.closest("[data-select-all]");
    if (allBtn) {
      state.selectedProjectId = "";
      $("projects-pane")?.classList.remove("mobile-open");
      refreshAll({ silent: true });
      return;
    }

    const runBtn = e.target.closest(".proj-run-btn");
    if (runBtn) {
      e.stopPropagation();
      handleRunBatch(runBtn.dataset.projectId);
      return;
    }

    const graveBtn = e.target.closest(".proj-graveyard-btn");
    if (graveBtn) {
      e.stopPropagation();
      openGraveyardModal(graveBtn.dataset.projectId);
      return;
    }

    if (e.target.closest(".toggle-label")) {
      e.stopPropagation();
      return;
    }

    const card = e.target.closest(".project-card");
    if (card) {
      state.selectedProjectId = card.dataset.projectId;
      $("projects-pane")?.classList.remove("mobile-open");
      refreshAll({ silent: true });
    }
  });

  $("projects-list")?.addEventListener("change", (e) => {
    const toggle = e.target.closest(".proj-continuous-toggle");
    if (toggle) {
      handleToggleContinuous(toggle.dataset.projectId, toggle.checked);
    }
  });

  // Status filter pills
  $("status-filter-bar")?.addEventListener("click", (e) => {
    const pill = e.target.closest(".filter-pill");
    if (!pill) return;
    state.statusFilter = pill.dataset.status;
    $("status-filter-bar").querySelectorAll(".filter-pill").forEach((b) => {
      b.classList.toggle("active", b === pill);
    });
    renderCandidatesList();
  });

  // Candidate search
  $("candidate-search")?.addEventListener("input", (e) => {
    state.searchQuery = e.target.value;
    renderCandidatesList();
  });

  // Candidate card click
  $("candidates-list")?.addEventListener("click", (e) => {
    const card = e.target.closest(".cand-card");
    if (card) {
      loadCandidateDetail(card.dataset.candId);
    }
  });

  // Inspector tabs
  document.querySelectorAll(".inspector-tabs .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });

  // Human actions
  $("act-keep")?.addEventListener("click", handleKeepAction);
  $("act-steer")?.addEventListener("click", openSteerModal);
  $("act-drop")?.addEventListener("click", openDropModal);
  $("act-export")?.addEventListener("click", handleExportAction);
  $("act-merge")?.addEventListener("click", openMergeModal);

  // SpecAgent Drawer
  $("btn-agent-drawer")?.addEventListener("click", openAgentDrawer);
  $("btn-agent-close")?.addEventListener("click", closeAgentDrawer);
  $("agent-backdrop")?.addEventListener("click", closeAgentDrawer);
  $("btn-agent-reconnect")?.addEventListener("click", () => connectAgentWs(true));
  $("btn-agent-stop")?.addEventListener("click", () => {
    if (state.agentWs && state.agentWs.readyState === WebSocket.OPEN) {
      state.agentWs.send(
        JSON.stringify({
          type: "stop",
          session_key: state.agentSessionKey || undefined,
        }),
      );
    }
  });

  $("agent-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("agent-input");
    const txt = input.value;
    if (!txt.trim()) return;
    input.value = "";
    sendAgentMessage(txt);
  });

  $("agent-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("agent-form")?.requestSubmit();
    }
  });

  // Keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("modal-backdrop").classList.contains("hidden")) {
        closeModal();
      } else if (state.drawerOpen) {
        closeAgentDrawer();
      }
      return;
    }

    const tag = document.activeElement?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (e.key.toLowerCase() === "r") {
      e.preventDefault();
      refreshAll();
    } else if (e.key === "j" || e.key === "k") {
      const list = getFilteredCandidates();
      if (!list.length) return;
      const idx = list.findIndex((c) => c.cand_id === state.selectedCandId);
      const nextIdx =
        e.key === "j"
          ? idx < 0
            ? 0
            : Math.min(list.length - 1, idx + 1)
          : idx <= 0
          ? 0
          : idx - 1;
      loadCandidateDetail(list[nextIdx].cand_id);
    } else if (state.candidateDetail && ["1", "2", "3", "4"].includes(e.key)) {
      const tabs = ["report", "diff", "evidence", "preview"];
      setActiveTab(tabs[Number(e.key) - 1]);
    }
  });
}

function startPolling() {
  window.setInterval(() => {
    if (!document.hidden && !state.authModalOpen && !state.authRequired) {
      refreshAll({ silent: true });
    }
  }, 15_000);
}

applyTheme();
updateAuthButton();
bindEvents();
refreshAll().then(startPolling);
