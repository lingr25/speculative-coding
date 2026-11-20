"""Structured tool outputs and model-visible output bounding."""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MAX_TOOL_BYTES = 50_000
DEFAULT_MAX_TOOL_LINES = 3_000
DEFAULT_MAX_TOOL_LINE_BYTES = 512
_TRUNC_HEAD_RATIO = 0.7


def safe_output_name(name: Any, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "output")).strip("._")
    return (cleaned or "output")[:limit]


def _clip_utf8(text: str, max_bytes: int) -> tuple[str, int]:
    raw = text.encode("utf-8")
    if max_bytes <= 0 or len(raw) <= max_bytes:
        return text, 0
    clipped = raw[:max_bytes].decode("utf-8", errors="ignore")
    return clipped + "…", len(raw) - len(clipped.encode("utf-8"))


def _shrink_json_value(value: Any, spill_ref: str, depth: int = 0) -> Any:
    if isinstance(value, str):
        raw = value.encode("utf-8")
        if len(raw) > 2_000:
            return {
                "truncated": True,
                "bytes": len(raw),
                "preview": value[:500],
                "spill_ref": spill_ref,
            }
        return value
    if isinstance(value, list):
        if depth >= 2:
            return {"truncated": True, "items": len(value), "spill_ref": spill_ref}
        shrunk = [_shrink_json_value(item, spill_ref, depth + 1) for item in value[:20]]
        if len(value) > 20:
            shrunk.append({"_more_items": len(value) - 20, "spill_ref": spill_ref})
        return shrunk
    if isinstance(value, dict):
        if depth >= 3:
            return {"truncated": True, "keys": list(value)[:10], "spill_ref": spill_ref}
        return {
            str(key): _shrink_json_value(item, spill_ref, depth + 1)
            for key, item in list(value.items())[:30]
        }
    return value


def bound_tool_output(
    text: str,
    *,
    max_bytes: int = DEFAULT_MAX_TOOL_BYTES,
    max_lines: int = DEFAULT_MAX_TOOL_LINES,
    max_line_bytes: int = DEFAULT_MAX_TOOL_LINE_BYTES,
    spill_path: Path | None = None,
    spill_ref: str = "",
    json_mode: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Bound model-visible tool text while preserving JSON structure when possible."""
    raw_text = "" if text is None else str(text)
    total_bytes = len(raw_text.encode("utf-8"))
    raw_lines = raw_text.split("\n") if raw_text else []
    total_lines = len(raw_lines)
    full_ref = spill_ref if spill_path is not None else ""

    over_budget = total_lines > max_lines or total_bytes > max_bytes
    if not over_budget:
        if max_line_bytes <= 0:
            return raw_text, {
                "truncated": False,
                "total_bytes": total_bytes,
                "total_lines": total_lines,
                "output_bytes": total_bytes,
                "output_lines": total_lines,
                "elided_bytes": 0,
                "column_truncated_lines": 0,
                "spill_ref": "",
            }
        clipped_lines = []
        column_truncated = 0
        for line in raw_lines:
            clipped, dropped = _clip_utf8(line, max_line_bytes)
            if dropped:
                column_truncated += 1
            clipped_lines.append(clipped)
        if column_truncated == 0:
            return raw_text, {
                "truncated": False,
                "total_bytes": total_bytes,
                "total_lines": total_lines,
                "output_bytes": total_bytes,
                "output_lines": total_lines,
                "elided_bytes": 0,
                "column_truncated_lines": 0,
                "spill_ref": "",
            }
        if spill_path is not None:
            spill_path.parent.mkdir(parents=True, exist_ok=True)
            spill_path.write_text(raw_text, encoding="utf-8")
        inline = "\n".join(clipped_lines)
        inline += f"\n...[COLUMN TRUNCATED: {column_truncated} lines over {max_line_bytes} bytes]..."
        return inline, {
            "truncated": True,
            "total_bytes": total_bytes,
            "total_lines": total_lines,
            "output_bytes": len(inline.encode("utf-8")),
            "output_lines": len(clipped_lines),
            "elided_bytes": 0,
            "column_truncated_lines": column_truncated,
            "spill_ref": full_ref,
        }

    if json_mode:
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None
        if parsed is not None:
            if spill_path is not None:
                spill_path.parent.mkdir(parents=True, exist_ok=True)
                spill_path.write_text(raw_text, encoding="utf-8")
            shrunk = _shrink_json_value(parsed, full_ref)
            inline = json.dumps(shrunk, ensure_ascii=False)
            if len(inline.encode("utf-8")) > max_bytes:
                inline = json.dumps(
                    {
                        "truncated": True,
                        "total_bytes": total_bytes,
                        "total_lines": total_lines,
                        "spill_ref": full_ref,
                        "note": "Full output spilled; inspect selectively before relying on details.",
                    },
                    ensure_ascii=False,
                )
            return inline, {
                "truncated": True,
                "total_bytes": total_bytes,
                "total_lines": total_lines,
                "output_bytes": len(inline.encode("utf-8")),
                "output_lines": 1,
                "elided_bytes": max(0, total_bytes - len(inline.encode("utf-8"))),
                "column_truncated_lines": 0,
                "spill_ref": full_ref,
            }

    if spill_path is not None:
        spill_path.parent.mkdir(parents=True, exist_ok=True)
        spill_path.write_text(raw_text, encoding="utf-8")

    clipped_lines = []
    column_truncated = 0
    for line in raw_lines:
        clipped, dropped = _clip_utf8(line, max_line_bytes) if max_line_bytes > 0 else (line, 0)
        if dropped:
            column_truncated += 1
        clipped_lines.append(clipped)

    head_budget = max(1, int(max_bytes * _TRUNC_HEAD_RATIO))
    tail_budget = max(1, max_bytes - head_budget)
    head_lines: list[str] = []
    head_bytes = 0
    head_end = 0
    for idx, line in enumerate(clipped_lines):
        size = len(line.encode("utf-8")) + 1
        if head_lines and head_bytes + size > head_budget:
            break
        head_lines.append(line)
        head_bytes += size
        head_end = idx + 1

    tail_lines: list[str] = []
    tail_bytes = 0
    tail_start = total_lines
    for idx in range(total_lines - 1, head_end - 1, -1):
        size = len(clipped_lines[idx].encode("utf-8")) + 1
        if tail_lines and tail_bytes + size > tail_budget:
            break
        tail_lines.insert(0, clipped_lines[idx])
        tail_bytes += size
        tail_start = idx
    if tail_start <= head_end:
        tail_lines = []
        tail_bytes = 0

    kept_bytes = head_bytes + tail_bytes
    elided_bytes = max(0, total_bytes - kept_bytes)
    marker = (
        f"\n...[TRUNCATED: {total_lines} lines / {total_bytes} bytes; "
        f"elided ~{elided_bytes} bytes; full output: {full_ref or 'not saved'}]"
        "...\n"
    )
    inline = "\n".join(head_lines) + marker + "\n".join(tail_lines)
    if len(inline.encode("utf-8")) > max_bytes + 1024:
        inline = "\n".join(head_lines)[: max(1, max_bytes // 2)] + marker
    return inline, {
        "truncated": True,
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "output_bytes": len(inline.encode("utf-8")),
        "output_lines": len(head_lines) + len(tail_lines),
        "elided_bytes": elided_bytes,
        "column_truncated_lines": column_truncated,
        "spill_ref": full_ref,
    }


@dataclass
class ToolResult:
    text: str = ""
    images: list[str] = field(default_factory=list)
    files: list[dict[str, str]] = field(default_factory=list)
    context_images: list[str] = field(default_factory=list)
    terminate_loop: bool = False
    return_value: Any = None


def context_image_parts(paths: list[str], limit: int = 4) -> list[dict[str, Any]]:
    """Verify and encode local images into base64 data URLs when Pillow is installed."""
    if not paths or limit <= 0:
        return []
    try:
        from PIL import Image
    except ImportError:
        return []
    parts: list[dict[str, Any]] = []
    for path in paths[: max(0, limit)]:
        try:
            with Path(path).open("rb") as file:
                raw = file.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                continue
            with Image.open(io.BytesIO(raw)) as picture:
                if picture.format not in ("PNG", "JPEG", "WEBP") or picture.width * picture.height > 16_000_000:
                    continue
                mime = Image.MIME[picture.format]
                picture.verify()
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")},
                }
            )
        except Exception:
            continue
    return parts


def normalize_tool_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, tuple) and len(value) in (2, 3):
        return ToolResult(str(value[0]), list(value[1] or []), list(value[2] or []) if len(value) == 3 else [])
    if isinstance(value, (dict, list)):
        return ToolResult(json.dumps(value, ensure_ascii=False))
    return ToolResult(str(value))
