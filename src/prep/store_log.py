from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable


def load_keys(
    path: Path,
    fields: Iterable[str] = ("user_id", "session_id"),
) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()

    if not path.exists() or path.stat().st_size == 0:
        return keys

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            keys.add(tuple(row.get(field, "") or "" for field in fields))

    return keys


def append_row(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(row)
    row_fields = list(row.keys())

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=row_fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        return

    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        old_fields = list(reader.fieldnames or [])

        if not old_fields:
            raise ValueError(f"CSV has no header: {path}")

        new_fields = [field for field in row_fields if field not in old_fields]

        if not new_fields:
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=old_fields,
                    extrasaction="ignore",
                )
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            return

        fieldnames = old_fields + new_fields
        temp_path = path.with_name(f"{path.name}.tmp")

        try:
            with temp_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()

                for old_row in reader:
                    writer.writerow(old_row)

                writer.writerow(row)
                target.flush()
                os.fsync(target.fileno())

            os.replace(temp_path, path)

        finally:
            if temp_path.exists():
                temp_path.unlink()
