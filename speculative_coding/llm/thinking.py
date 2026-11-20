"""Preserved Thinking extraction and streaming `<think>...</think>` tag splitter."""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_think_and_clean(text: str, existing_reasoning: str = "") -> tuple[str, str]:
    """Losslessly split `(reasoning_text, clean_text)` from text containing `<think>...</think>`."""
    thoughts: list[str] = []
    if existing_reasoning and existing_reasoning.strip():
        thoughts.append(existing_reasoning.strip())
    if not text:
        return "\n\n".join(thoughts).strip(), ""
    for match in _THINK_RE.finditer(text):
        chunk = match.group(1).strip()
        if chunk:
            thoughts.append(chunk)
    cleaned = _THINK_RE.sub("", text)
    lower = cleaned.lower()
    if "<think>" in lower:
        idx = lower.find("<think>")
        dangling = cleaned[idx + 7 :].strip()
        if dangling:
            thoughts.append(dangling)
        cleaned = cleaned[:idx]
    return "\n\n".join(thoughts).strip(), cleaned.strip()


def _strip_think_tags(text: str) -> str:
    _, cleaned = extract_think_and_clean(text)
    return cleaned


class TagStreamSplitter:
    """Sliding-buffer `<think>...</think>` streaming splitter preventing tag tearing across chunks."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self.in_think: bool = False
        self._buf: str = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Feed a streaming text chunk and return `[('reasoning' | 'text', delta_str), ...]`."""
        if not chunk:
            return []
        self._buf += chunk
        out: list[tuple[str, str]] = []

        while self._buf:
            target = self.CLOSE_TAG if self.in_think else self.OPEN_TAG
            lower_buf = self._buf.lower()
            idx = lower_buf.find(target)
            if idx != -1:
                if idx > 0:
                    kind = "reasoning" if self.in_think else "text"
                    out.append((kind, self._buf[:idx]))
                self._buf = self._buf[idx + len(target) :]
                self.in_think = not self.in_think
                continue

            keep = 0
            max_check = min(len(self._buf), len(target) - 1)
            for k in range(max_check, 0, -1):
                if target.startswith(lower_buf[-k:]):
                    keep = k
                    break

            emit_len = len(self._buf) - keep
            if emit_len > 0:
                kind = "reasoning" if self.in_think else "text"
                out.append((kind, self._buf[:emit_len]))
                self._buf = self._buf[emit_len:]
            break

        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        kind = "reasoning" if self.in_think else "text"
        rem = self._buf
        self._buf = ""
        return [(kind, rem)]
