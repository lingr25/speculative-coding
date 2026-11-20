"""Token estimation, tool schema validation, orphan repair, and atomic context compaction."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("speculative_coding.llm.context")

_TOOL_SCHEMA_VALIDATOR_WARNED = False


def env_token_budget(name: str, default: int = 0) -> int | None:
    """Configurable token fuse; ``0`` (default) disables the per-turn limit."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = default
    else:
        value = default
    return value if value > 0 else None


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Zero-dependency fast token estimator used when providers omit usage counts."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            cjk = sum(1 for ch in content if "\u4e00" <= ch <= "\u9fff")
            other = len(content) - cjk
            total += int(cjk * 0.65 + other * 0.28) + 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        txt = part.get("text") or ""
                        total += int(len(txt) * 0.4)
                    elif part.get("type") == "image_url":
                        total += 1000
        if m.get("reasoning_content"):
            total += int(len(str(m["reasoning_content"])) * 0.4)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += int(len(str(fn.get("name") or "") + str(fn.get("arguments") or "")) * 0.35) + 8
    return max(1, total)


def estimate_tools_tokens(tools: list[dict[str, Any]] | None) -> int:
    """Estimate the prompt cost of tool schemas sent on every step."""
    if not tools:
        return 0
    try:
        return max(1, int(len(json.dumps(tools, ensure_ascii=False)) * 0.3))
    except Exception:
        return 0


def validate_tool_arguments(
    name: str,
    arguments: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
) -> str:
    """Return a short validation error, or an empty string when the call is valid."""
    schema = schemas.get(name)
    if not isinstance(schema, dict) or not schema:
        return ""
    global _TOOL_SCHEMA_VALIDATOR_WARNED
    try:
        import jsonschema
    except ImportError:
        if not _TOOL_SCHEMA_VALIDATOR_WARNED:
            logger.warning("jsonschema not installed; skipping tool argument schema validation")
            _TOOL_SCHEMA_VALIDATOR_WARNED = True
        return ""
    try:
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.path))
    except Exception as exc:
        logger.debug("Skipping schema validation for tool %s: %s", name, exc)
        return ""
    if not errors:
        return ""
    return "; ".join(
        f"{'/'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in errors[:3]
    )


def _merge_user_with_checkpoint(
    user_msg: dict[str, Any],
    checkpoint_text: str,
) -> dict[str, Any]:
    """Combine a trigger user message and a compaction checkpoint into one user message."""
    content = user_msg.get("content")
    if isinstance(content, list):
        merged_parts = [
            part
            for part in content
            if not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and str(part.get("text") or "").lstrip().startswith("[Auto-Compact Checkpoint")
            )
        ]
        merged_parts.append({"type": "text", "text": "\n\n" + checkpoint_text})
        return {**user_msg, "role": "user", "content": merged_parts}
    base_text = str(content or "").strip()
    combined = f"{base_text}\n\n{checkpoint_text}" if base_text else checkpoint_text
    return {**user_msg, "role": "user", "content": combined}


def emergency_compact(
    messages: list[dict[str, Any]],
    keep_recent_groups: int = 2,
    max_tool_chars: int = 2_000,
) -> list[dict[str, Any]]:
    """Guaranteed size reduction for a provider that already rejected the prompt."""
    if not messages:
        return messages
    has_system = messages[0].get("role") == "system"
    system = messages[:1] if has_system else []
    rest = messages[1:] if has_system else list(messages)

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in rest:
        if message.get("role") == "assistant":
            if current:
                groups.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        groups.append(current)

    keep = max(0, keep_recent_groups)
    kept = groups[-keep:] if keep else []
    kept_ids = {id(message) for group in kept for message in group}
    trigger: dict[str, Any] | None = None
    for message in reversed(rest):
        if message.get("role") == "user" and id(message) not in kept_ids:
            content = message.get("content")
            if isinstance(content, list):
                trigger = message
                break
            text = str(content or "")
            if not text.startswith("[Auto-Compact Checkpoint"):
                if "\n\n[Auto-Compact Checkpoint" in text:
                    clean_text = text.split("\n\n[Auto-Compact Checkpoint", 1)[0].strip()
                    trigger = {**message, "content": clean_text}
                else:
                    trigger = message
                break

    marker_text = (
        "[Auto-Compact Checkpoint (Emergency Overflow Compact)]: "
        "Earlier tool steps were omitted due to context window overflow; "
        "continue from the preserved recent steps without repeating completed work."
    )
    out = list(system)
    if trigger is not None:
        out.append(_merge_user_with_checkpoint(trigger, marker_text))
    else:
        out.append({"role": "user", "content": marker_text})
    for group in kept:
        for message in group:
            content = message.get("content")
            if message.get("role") == "tool" and isinstance(content, str) and len(content) > max_tool_chars:
                message = {
                    **message,
                    "content": content[:max_tool_chars]
                    + "\n...[tool output truncated during overflow recovery]...",
                }
            out.append(message)
    return out


def repair_orphaned_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair unclosed orphaned tool_calls by inserting tombstone tool results."""
    repaired: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        msg = messages[i]
        repaired.append(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = [
                tc.get("id") for tc in (msg.get("tool_calls") or []) if isinstance(tc, dict) and tc.get("id")
            ]
            expected_names = {
                tc.get("id"): (tc.get("function") or {}).get("name", "unknown")
                for tc in (msg.get("tool_calls") or [])
                if isinstance(tc, dict) and tc.get("id")
            }
            found_ids: set[str] = set()
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                tid = messages[j].get("tool_call_id")
                if tid:
                    found_ids.add(tid)
                repaired.append(messages[j])
                j += 1
            for eid in expected_ids:
                if eid not in found_ids:
                    repaired.append(
                        {
                            "role": "tool",
                            "tool_call_id": eid,
                            "name": expected_names.get(eid, "unknown"),
                            "content": json.dumps(
                                {
                                    "status": "interrupted",
                                    "reason": "Turn was cancelled or interrupted before tool completion",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
            i = j
        else:
            i += 1
    return repaired


def compact_in_loop_atomic(
    messages: list[dict[str, Any]],
    keep_recent_groups: int = 6,
    **_kwargs: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Safely compact along Atomic Tool Group boundaries without breaking tool_call pairing."""
    if len(messages) <= 4:
        return messages, 0

    has_system = bool(messages and messages[0].get("role") == "system")
    sys_prefix = messages[:1] if has_system else []
    start_idx = 1 if has_system else 0

    prefix_end = len(messages)
    for idx in range(start_idx, len(messages)):
        m = messages[idx]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            prefix_end = idx
            break

    raw_prefix = messages[start_idx:prefix_end]
    tail = messages[prefix_end:]

    groups: list[list[dict[str, Any]]] = []
    cur_group: list[dict[str, Any]] = []
    for m in tail:
        role = m.get("role")
        if role == "assistant":
            if cur_group:
                groups.append(cur_group)
            cur_group = [m]
        else:
            cur_group.append(m)
    if cur_group:
        groups.append(cur_group)

    compact_header = "[Auto-Compact Checkpoint (In-Loop)]:"
    prior_summary_lines: list[str] = []
    kept_trigger_user: dict[str, Any] | None = None
    evicted_prefix_count = 0

    if raw_prefix:
        for idx in range(len(raw_prefix) - 1, -1, -1):
            pm = raw_prefix[idx]
            content = pm.get("content")
            if pm.get("role") == "user" and isinstance(content, list):
                embedded_summary_parts = [
                    str(part.get("text") or "").lstrip()
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and str(part.get("text") or "").lstrip().startswith(compact_header)
                ]
                for emb in embedded_summary_parts:
                    for ln in emb.splitlines()[1:]:
                        if ln.strip():
                            prior_summary_lines.append(ln.strip())
                    evicted_prefix_count += 1
                kept_trigger_user = pm
                older_prefix = raw_prefix[:idx] + raw_prefix[idx + 1 :]
                break
            c_str = str(content or "")
            if pm.get("role") == "user" and not c_str.startswith(compact_header):
                if compact_header in c_str:
                    base_part, summary_part = c_str.split(compact_header, 1)
                    kept_trigger_user = {**pm, "content": base_part.rstrip()}
                    for ln in summary_part.splitlines():
                        if ln.strip():
                            prior_summary_lines.append(ln.strip())
                    evicted_prefix_count += 1
                else:
                    kept_trigger_user = pm
                older_prefix = raw_prefix[:idx] + raw_prefix[idx + 1 :]
                break
        else:
            older_prefix = raw_prefix

        for pm in older_prefix:
            c_str = str(pm.get("content") or "")
            if c_str.startswith(compact_header):
                for ln in c_str.splitlines()[1:]:
                    if ln.strip():
                        prior_summary_lines.append(ln.strip())
                evicted_prefix_count += 1
            else:
                role_label = pm.get("role", "msg")
                snippet = c_str[:180].replace("\n", " ").strip()
                if snippet:
                    prior_summary_lines.append(f"- History ({role_label}): {snippet}")
                evicted_prefix_count += 1

    evict_groups: list[list[dict[str, Any]]] = []
    kept_groups = groups
    if len(groups) > keep_recent_groups:
        evict_groups = groups[:-keep_recent_groups]
        kept_groups = groups[-keep_recent_groups:]

    total_evicted = len(evict_groups) + evicted_prefix_count
    if total_evicted == 0:
        return messages, 0

    summary_lines: list[str] = [compact_header]
    if prior_summary_lines:
        summary_lines.extend(prior_summary_lines[-16:])

    for g_idx, grp in enumerate(evict_groups, 1):
        a_msg = grp[0]
        tcs = a_msg.get("tool_calls") or []
        names = [str((tc.get("function") or {}).get("name") or "?") for tc in tcs]
        tool_previews: list[str] = []
        for tm in grp[1:]:
            if tm.get("role") == "tool":
                c = str(tm.get("content") or "")[:180].replace("\n", " ")
                tool_previews.append(f"{tm.get('name', 'tool')}->{c}")
        summary_lines.append(
            f"- StepGroup #{g_idx}: called [{', '.join(names)}] | summary: {'; '.join(tool_previews)[:360]}"
        )

    if len(summary_lines) > 25:
        summary_lines = [summary_lines[0], *summary_lines[-24:]]

    checkpoint_text = "\n".join(summary_lines)

    new_messages: list[dict[str, Any]] = list(sys_prefix)
    if kept_trigger_user is not None:
        new_messages.append(_merge_user_with_checkpoint(kept_trigger_user, checkpoint_text))
    else:
        new_messages.append({"role": "user", "content": checkpoint_text})
    for grp in kept_groups:
        new_messages.extend(grp)

    if estimate_tokens(new_messages) >= estimate_tokens(messages) and len(evict_groups) == 0:
        return messages, 0
    return new_messages, max(1, len(evict_groups) or evicted_prefix_count)
