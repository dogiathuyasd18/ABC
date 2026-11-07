"""Append-only CSV logs with a stable header."""

from __future__ import annotations

import csv
from pathlib import Path


def load_keys(path: Path, fields=("user_id", "session_id")) -> set[tuple]:
    keys = set()
    if not path.exists() or path.stat().st_size == 0:
        return keys
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add(tuple(row.get(field, "") for field in fields))
    return keys


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing = csv.DictReader(handle)
            if existing.fieldnames:
                fieldnames = list(existing.fieldnames)
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()
