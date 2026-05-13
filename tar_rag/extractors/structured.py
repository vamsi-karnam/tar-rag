"""Structured-data extractors (JSON, CSV).

These flatten the structure into readable text rather than preserving
machine schema. The output is consumed downstream by alias extraction and
``text_sample`` generation only — high fidelity is not required.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .base import TextExtractor


class JsonTextExtractor(TextExtractor):
    """Recursively flatten a JSON document into ``key: value`` lines.

    Lists are flattened as ``key[index]``. Non-string scalars are stringified.
    Binary-ish values (very large numbers, deeply nested objects) are kept
    as their ``str()`` form.
    """

    name = "JsonTextExtractor"
    _max_value_length = 2_000

    def extract(self, file_path: str) -> str:
        raw = Path(file_path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Treat malformed JSON as plaintext — better than dropping the
            # document entirely.
            return raw

        lines: list[str] = []
        self._walk(data, "", lines)
        return "\n".join(lines)

    def _walk(self, node: Any, prefix: str, lines: list[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                self._walk(value, child_prefix, lines)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                self._walk(value, child_prefix, lines)
        else:
            value_str = "" if node is None else str(node)
            if len(value_str) > self._max_value_length:
                value_str = value_str[: self._max_value_length] + "…"
            label = prefix or "value"
            lines.append(f"{label}: {value_str}")


class CsvTextExtractor(TextExtractor):
    """Render a CSV as ``header: value`` lines per row.

    For files without a recognisable header row we fall back to
    positional column labels (``col_0``, ``col_1``, …).
    """

    name = "CsvTextExtractor"
    _max_rows = 5_000

    def extract(self, file_path: str) -> str:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        reader = csv.reader(text.splitlines())
        rows = list(reader)
        if not rows:
            return ""

        header = rows[0]
        body = rows[1:] if self._looks_like_header(header) else rows
        if not self._looks_like_header(header):
            header = [f"col_{index}" for index in range(len(header))]

        lines: list[str] = []
        for row_index, row in enumerate(body[: self._max_rows]):
            cells = []
            for column_index, value in enumerate(row):
                name = header[column_index] if column_index < len(header) else f"col_{column_index}"
                cells.append(f"{name}: {value}")
            lines.append(f"row {row_index + 1}: " + " | ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _looks_like_header(row: list[str]) -> bool:
        if not row:
            return False
        for cell in row:
            cell_str = cell.strip()
            if not cell_str:
                return False
            # If a cell is numeric, this row probably isn't a header.
            try:
                float(cell_str)
                return False
            except ValueError:
                continue
        return True
