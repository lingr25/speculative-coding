/**
 * Pure, XSS-safe rendering & formatting utilities for Speculative Coding Review Console.
 * Zero external dependencies — works completely offline.
 */

const HTML_ESCAPE_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => HTML_ESCAPE_MAP[ch]);
}

export function shortSha(sha) {
  const s = String(sha || "").trim();
  return s ? s.slice(0, 8) : "—";
}

export function formatTokens(n) {
  const num = Number(n || 0);
  if (!Number.isFinite(num) || num <= 0) return "0";
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
  return String(Math.round(num));
}

export function formatBytes(bytes) {
  const num = Number(bytes || 0);
  if (!Number.isFinite(num) || num <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let idx = 0;
  let val = num;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx += 1;
  }
  return `${val >= 100 || idx === 0 ? val.toFixed(0) : val.toFixed(1)} ${units[idx]}`;
}

export function formatTimestamp(ts) {
  if (!ts) return "—";
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatRelativeTime(ts) {
  if (!ts) return "—";
  const diffSec = Math.floor(Date.now() / 1000 - Number(ts));
  if (!Number.isFinite(diffSec)) return "—";
  if (diffSec < 10) return "just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return `${Math.floor(diffSec / 86400)}d ago`;
}

export function formatTtl(expiresAt, status) {
  if (status === "kept") return { text: "TTL Frozen", cls: "badge-success" };
  if (status === "merged") return { text: "Merged", cls: "badge-purple" };
  if (!expiresAt) return null;
  const remSec = Math.floor(Number(expiresAt) - Date.now() / 1000);
  if (!Number.isFinite(remSec)) return null;
  if (remSec <= 0) return { text: "TTL Expiring", cls: "badge-danger" };
  const hours = Math.floor(remSec / 3600);
  const mins = Math.floor((remSec % 3600) / 60);
  const cls = hours < 6 ? "badge-warning" : "badge-muted";
  if (hours >= 24) {
    const days = Math.floor(hours / 24);
    const remH = hours % 24;
    return { text: `TTL ${days}d ${remH}h`, cls };
  }
  return { text: `TTL ${hours}h ${mins}m`, cls };
}

export function statusBadgeClass(status) {
  switch (status) {
    case "staged":
      return "badge-info";
    case "kept":
      return "badge-success";
    case "merged":
      return "badge-purple";
    case "coding":
    case "revising":
    case "building":
    case "reviewing":
    case "inspecting":
    case "dehydrating":
      return "badge-warning";
    case "dropped":
    case "expired":
    case "error":
      return "badge-danger";
    default:
      return "badge-muted";
  }
}

export function getModifiedTests(candidate) {
  if (!candidate) return [];
  const raw =
    candidate.modified_existing_tests ??
    candidate.modified_existing_tests_json ??
    candidate.preview_meta_json?.modified_existing_tests ??
    [];
  if (Array.isArray(raw)) return raw.filter((item) => typeof item === "string" && item.trim());
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string" && item.trim()) : [];
    } catch {
      return [];
    }
  }
  return [];
}

/* ==========================================================================
   Safe Markdown Renderer (Zero Raw HTML Passthrough / Anti-XSS)
   ========================================================================== */

function sanitizeUrl(rawUrl) {
  const trimmed = String(rawUrl || "").trim();
  if (/^https?:\/\/[^\s<>"']+$/i.test(trimmed)) {
    return trimmed;
  }
  return "";
}

function renderInlineMarkdown(rawText) {
  // Tokenize inline code spans first so markdown symbols inside `code` are untouched.
  const segments = [];
  const codeRe = /`([^`\n]+)`/g;
  let lastIndex = 0;
  let match;
  while ((match = codeRe.exec(rawText)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: "text", value: rawText.slice(lastIndex, match.index) });
    }
    segments.push({ type: "code", value: match[1] });
    lastIndex = codeRe.lastIndex;
  }
  if (lastIndex < rawText.length) {
    segments.push({ type: "text", value: rawText.slice(lastIndex) });
  }

  return segments
    .map((seg) => {
      if (seg.type === "code") {
        return `<code>${escapeHtml(seg.value)}</code>`;
      }
      // Escape HTML first! Every tag produced below is strictly controlled.
      let html = escapeHtml(seg.value);
      // Safe links: [label](https://...) — url is already HTML-escaped from escapeHtml(seg.value)
      html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
        const safe = sanitizeUrl(url);
        if (!safe) return `${label} (${url})`;
        return `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>`;
      });
      // Bold **text**
      html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
      // Strikethrough ~~text~~
      html = html.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
      // Italic *text*
      html = html.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");
      return html;
    })
    .join("");
}

function isTableSeparator(line) {
  return /^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line);
}

function splitTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

export function renderMarkdown(markdown) {
  const src = String(markdown || "").replace(/\r\n/g, "\n").trim();
  if (!src) {
    return `<p class="empty-inline">No report available yet.</p>`;
  }

  const lines = src.split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Blank line
    if (!line.trim()) {
      i += 1;
      continue;
    }

    // Fenced code block
    const fenceMatch = line.match(/^```([a-zA-Z0-9_+-]*)\s*$/);
    if (fenceMatch) {
      const lang = escapeHtml(fenceMatch[1] || "");
      const codeLines = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      const cls = lang ? ` class="language-${lang}"` : "";
      out.push(`<pre><code${cls}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      out.push(`<h${level}>${renderInlineMarkdown(headingMatch[2].trim())}</h${level}>`);
      i += 1;
      continue;
    }

    // Horizontal rule
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
      out.push("<hr />");
      i += 1;
      continue;
    }

    // Pipe Table
    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = splitTableRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
        rows.push(splitTableRow(lines[i]));
        i += 1;
      }
      const thead = `<thead><tr>${headers.map((h) => `<th>${renderInlineMarkdown(h)}</th>`).join("")}</tr></thead>`;
      const tbody = `<tbody>${rows
        .map((r) => `<tr>${headers.map((_, idx) => `<td>${renderInlineMarkdown(r[idx] || "")}</td>`).join("")}</tr>`)
        .join("")}</tbody>`;
      out.push(`<table>${thead}${tbody}</table>`);
      continue;
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const quoteLines = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      out.push(`<blockquote>${quoteLines.map((l) => renderInlineMarkdown(l)).join("<br />")}</blockquote>`);
      continue;
    }

    // Unordered list
    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
        i += 1;
      }
      out.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i += 1;
      }
      out.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
      continue;
    }

    // Paragraph
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^```/.test(lines[i]) &&
      !/^#{1,4}\s+/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !/^\s*>\s?/.test(lines[i]) &&
      !/^\s*([-*_])\1{2,}\s*$/.test(lines[i])
    ) {
      paraLines.push(lines[i]);
      i += 1;
    }
    out.push(`<p>${paraLines.map((l) => renderInlineMarkdown(l)).join("<br />")}</p>`);
  }

  return out.join("\n");
}

/* ==========================================================================
   Unified Diff Parser & Colorized Renderer
   ========================================================================== */

export function parseUnifiedDiff(diffText) {
  const raw = String(diffText || "").replace(/\r\n/g, "\n");
  if (!raw.trim()) return [];

  const lines = raw.split("\n");
  const files = [];
  let current = null;
  let oldLine = 0;
  let newLine = 0;

  const ensureFile = (name) => {
    current = {
      id: `diff-file-${files.length}`,
      name: name || `file-${files.length + 1}`,
      additions: 0,
      deletions: 0,
      rows: [],
    };
    files.push(current);
  };

  for (const line of lines) {
    if (line.startsWith("diff --git ")) {
      const m = line.match(/^diff --git a\/(.+?) b\/(.+)$/);
      ensureFile(m ? m[2] : line.slice("diff --git ".length));
      current.rows.push({ kind: "meta", oldLn: "", newLn: "", text: line });
      continue;
    }

    if (!current) {
      if (line.startsWith("--- ") || line.startsWith("+++ ")) {
        ensureFile(line.slice(4).replace(/^[ab]\//, ""));
      } else {
        ensureFile("patch");
      }
    }

    if (line.startsWith("+++ ")) {
      const path = line.slice(4).trim().replace(/^b\//, "");
      if (path && path !== "/dev/null") {
        current.name = path;
      }
      current.rows.push({ kind: "meta", oldLn: "", newLn: "", text: line });
    } else if (
      line.startsWith("--- ") ||
      line.startsWith("index ") ||
      line.startsWith("new file mode") ||
      line.startsWith("deleted file mode") ||
      line.startsWith("rename ") ||
      line.startsWith("similarity ") ||
      line.startsWith("Binary files ")
    ) {
      current.rows.push({ kind: "meta", oldLn: "", newLn: "", text: line });
    } else if (line.startsWith("@@")) {
      const m = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      if (m) {
        oldLine = Number(m[1]);
        newLine = Number(m[2]);
      }
      current.rows.push({ kind: "hunk", oldLn: "…", newLn: "…", text: line });
    } else if (line.startsWith("+")) {
      current.additions += 1;
      current.rows.push({ kind: "add", oldLn: "", newLn: String(newLine++), text: line });
    } else if (line.startsWith("-")) {
      current.deletions += 1;
      current.rows.push({ kind: "del", oldLn: String(oldLine++), newLn: "", text: line });
    } else {
      current.rows.push({
        kind: "ctx",
        oldLn: oldLine ? String(oldLine++) : "",
        newLn: newLine ? String(newLine++) : "",
        text: line,
      });
    }
  }

  return files;
}

export function renderDiffViewer(diffText, modifiedTests = []) {
  const files = parseUnifiedDiff(diffText);
  if (!files.length) {
    return {
      fileCount: 0,
      html: `<div class="empty-inline">No diff captured for this candidate yet.</div>`,
    };
  }

  const testSet = new Set(modifiedTests);
  const totalAdd = files.reduce((acc, f) => acc + f.additions, 0);
  const totalDel = files.reduce((acc, f) => acc + f.deletions, 0);

  const sidebarItems = files
    .map((f) => {
      const isTest = testSet.has(f.name);
      return `
        <button type="button" class="diff-file-jump ${isTest ? "is-test" : ""}" data-jump-target="${escapeHtml(f.id)}" title="${escapeHtml(f.name)}">
          <span class="diff-jump-name">${isTest ? "⚠ " : ""}${escapeHtml(f.name)}</span>
          <span class="diff-counts">
            <span class="cnt-add">+${f.additions}</span>
            <span class="cnt-del">-${f.deletions}</span>
          </span>
        </button>`;
    })
    .join("");

  const fileCards = files
    .map((f) => {
      const isTest = testSet.has(f.name);
      const rowsHtml = f.rows
        .map((row) => {
          const cls =
            row.kind === "add"
              ? "diff-row-add"
              : row.kind === "del"
              ? "diff-row-del"
              : row.kind === "hunk"
              ? "diff-row-hunk"
              : row.kind === "meta"
              ? "diff-row-meta"
              : "";
          return `<tr class="${cls}"><td class="diff-ln">${escapeHtml(row.oldLn)}</td><td class="diff-ln">${escapeHtml(row.newLn)}</td><td class="diff-code">${escapeHtml(row.text)}</td></tr>`;
        })
        .join("");

      return `
        <article class="diff-file-card ${isTest ? "test-modified" : ""}" id="${escapeHtml(f.id)}">
          <div class="diff-file-header">
            <span>${escapeHtml(f.name)} ${isTest ? `<span class="badge badge-warning">Existing Test Modified</span>` : ""}</span>
            <span class="diff-counts">
              <span class="cnt-add">+${f.additions}</span>
              <span class="cnt-del">-${f.deletions}</span>
            </span>
          </div>
          <div class="diff-table-wrap">
            <table class="diff-table"><tbody>${rowsHtml}</tbody></table>
          </div>
        </article>`;
    })
    .join("");

  const html = `
    <div class="diff-layout">
      <aside class="diff-sidebar">
        <div class="diff-sidebar-head">
          <span>${files.length} file${files.length === 1 ? "" : "s"} (+${totalAdd} / -${totalDel})</span>
          <button type="button" class="btn btn-ghost btn-xs" id="btn-copy-raw-diff">Copy</button>
        </div>
        <div class="diff-file-list">${sidebarItems}</div>
      </aside>
      <div class="diff-files">${fileCards}</div>
    </div>`;

  return { fileCount: files.length, html };
}

/* ==========================================================================
   Gates & Review Evidence Timeline Renderer
   ========================================================================== */

function renderGateItem(g) {
  const code = g?.code;
  const ok = code === 0;
  const output = String(g?.output || "").trim();
  return `
    <div class="gate-card">
      <div class="gate-head">
        <span class="gate-cmd">${escapeHtml(g?.command || "(command)")}</span>
        <span class="badge ${ok ? "badge-success" : "badge-danger"}">exit ${escapeHtml(code ?? "?")}</span>
      </div>
      ${output ? `<pre class="gate-output">${escapeHtml(output)}</pre>` : ""}
    </div>`;
}

export function renderEvidencePanel(candidate, events = []) {
  const meta = candidate?.preview_meta_json || {};
  const gates = Array.isArray(meta.gates) ? meta.gates : [];
  const regression = meta.baseline_test_regression;

  let gatesSection = "";
  if (gates.length) {
    gatesSection = `
      <div>
        <h4 class="section-heading">Deterministic Gates (${meta.passed ? "Passed" : "Recorded"})</h4>
        ${gates.map(renderGateItem).join("")}
      </div>`;
  } else {
    gatesSection = `
      <div>
        <h4 class="section-heading">Deterministic Gates</h4>
        <div class="empty-inline">No gate output stored on candidate yet.</div>
      </div>`;
  }

  let regressionSection = "";
  if (regression && regression.checked) {
    const regGates = Array.isArray(regression.gates) ? regression.gates : [];
    const passed = Boolean(regression.passed);
    regressionSection = `
      <div>
        <h4 class="section-heading">
          Baseline Test Regression Guard
          <span class="badge ${passed ? "badge-success" : "badge-danger"}">${passed ? "Baseline Passed" : "Baseline Failed"}</span>
        </h4>
        <div class="info-box" style="margin-bottom:8px;">
          Verified original <code>base_sha</code> tests against candidate's non-test changes for:
          ${(regression.modified_existing_tests || []).map((f) => `<span class="test-warning-file">${escapeHtml(f)}</span>`).join(" ")}
        </div>
        ${regGates.map(renderGateItem).join("")}
      </div>`;
  }

  const timelineHtml = events.length
    ? `<div class="timeline">
        ${events
          .map((ev) => {
            const payload = ev.payload_json && Object.keys(ev.payload_json).length ? ev.payload_json : null;
            const kind = String(ev.kind || "event");
            return `
              <div class="timeline-item kind-${escapeHtml(kind)}">
                <div class="timeline-head">
                  <div>
                    <span class="badge badge-muted">${escapeHtml(kind)}</span>
                    ${ev.head_sha ? `<span class="mono-subtle" style="margin-left:6px;">${escapeHtml(shortSha(ev.head_sha))}</span>` : ""}
                  </div>
                  <span class="mono-subtle">${escapeHtml(formatTimestamp(ev.created_at))}</span>
                </div>
                <div class="timeline-summary">${escapeHtml(ev.summary || "")}</div>
                ${
                  payload
                    ? `<details class="timeline-payload">
                         <summary class="mono-subtle">Payload details</summary>
                         <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
                       </details>`
                    : ""
                }
              </div>`;
          })
          .join("")}
      </div>`
    : `<div class="empty-inline">No recorded evidence events yet.</div>`;

  return `
    <div class="evidence-section">
      ${regressionSection}
      ${gatesSection}
      <div>
        <h4 class="section-heading">Audit &amp; Review Timeline (${events.length})</h4>
        ${timelineHtml}
      </div>
    </div>`;
}
