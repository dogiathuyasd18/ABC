#!/usr/bin/env python3
"""Audit the ABC head-pose preprocessing pipeline and produce Markdown/JSON reports.

Run from the repository root:
    python scripts/audit_pipeline.py
    python scripts/audit_pipeline.py --root . --max-gb 10 --reserve-free-gb 5

The auditor is deliberately conservative:
- PASS means the check was directly verified.
- WARN means evidence is incomplete or a manual review is still needed.
- FAIL means a concrete violation was detected.
- NOT_CHECKED means the repository's schema/implementation does not expose enough
  information for this script to verify the requirement safely.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("PASS", "WARN", "FAIL", "NOT_CHECKED")


class Audit:
    def __init__(self, root: Path, max_gb: float, reserve_free_gb: float):
        self.root = root.resolve()
        self.max_bytes = int(max_gb * 1024**3)
        self.reserve_bytes = int(reserve_free_gb * 1024**3)
        self.results: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str, evidence: Any = None):
        if status not in STATUSES:
            raise ValueError(status)
        self.results.append({
            "check": name,
            "status": status,
            "detail": detail,
            "evidence": evidence,
        })

    def path(self, value: str) -> Path:
        return self.root / value

    def check_required_files(self):
        required = [
            "src/prep/session.py",
            "src/prep/resampling.py",
            "src/prep/windows.py",
            "src/prep/store_log.py",
            "src/prep/download.py",
        ]
        missing = [x for x in required if not self.path(x).is_file()]
        if missing:
            self.add("Required preprocessing files", "FAIL",
                     "Missing expected files: " + ", ".join(missing))
        else:
            self.add("Required preprocessing files", "PASS",
                     "All expected preprocessing files exist.", required)

    def check_compile(self):
        src = self.path("src")
        if not src.exists():
            self.add("Python compilation", "FAIL", "src/ directory is missing.")
            return
        proc = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", str(src)],
            cwd=self.root, text=True, capture_output=True
        )
        if proc.returncode == 0:
            self.add("Python compilation", "PASS",
                     "compileall completed without syntax errors.")
        else:
            self.add("Python compilation", "FAIL",
                     (proc.stderr or proc.stdout).strip()[-5000:])

    def check_imports(self):
        modules = [
            "src.prep.config",
            "src.prep.filtering",
            "src.prep.cleaning",
            "src.prep.quality",
            "src.prep.resampling",
            "src.prep.windows",
            "src.prep.session",
            "src.prep.store_log",
            "src.prep.download",
        ]
        failures = []
        old_path = list(sys.path)
        try:
            sys.path.insert(0, str(self.root))
            for name in modules:
                try:
                    importlib.import_module(name)
                except Exception as exc:
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            sys.path[:] = old_path
        if failures:
            self.add("Preprocessing imports", "FAIL",
                     "One or more modules cannot be imported.",
                     failures)
        else:
            self.add("Preprocessing imports", "PASS",
                     "All expected preprocessing modules imported.")

    def check_config(self):
        cfg_path = self.path("src/prep/config.py")
        if not cfg_path.exists():
            self.add("Challenge filter configuration", "NOT_CHECKED",
                     "src/prep/config.py was not found.")
            return
        try:
            sys.path.insert(0, str(self.root))
            mod = importlib.import_module("src.prep.config")
            found = {}
            for cls_name in ("FilterConfig", "CleanConfig", "WindowConfig"):
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    try:
                        obj = cls()
                        found[cls_name] = vars(obj)
                    except Exception as exc:
                        found[cls_name] = f"Could not instantiate: {exc}"
            text = cfg_path.read_text(errors="replace").lower()
            checks = {
                "sampling-rate lower bound 85 Hz": any(
                    token in text for token in ("85", "min_sampling_rate", "min_sample_rate")
                ),
                "sampling-rate upper bound 110 Hz": any(
                    token in text for token in ("110", "max_sampling_rate", "max_sample_rate")
                ),
                "minimum sweep 17 degrees": "17" in text,
                "minimum velocity 15 degrees/second": "15" in text,
            }
            unresolved = [k for k, v in checks.items() if not v]
            if unresolved:
                self.add("Challenge filter configuration", "WARN",
                         "Could not locate all expected threshold indicators. "
                         "Inspect config and runtime filter logic manually.",
                         {"detected": checks, "default_configs": found})
            else:
                self.add("Challenge filter configuration", "WARN",
                         "Threshold-like values were found in config.py, but this "
                         "does not prove that the filtering code applies them correctly.",
                         {"detected": checks, "default_configs": found})
        except Exception as exc:
            self.add("Challenge filter configuration", "NOT_CHECKED",
                     f"Could not inspect configuration: {type(exc).__name__}: {exc}")
        finally:
            if sys.path and sys.path[0] == str(self.root):
                sys.path.pop(0)

    def _csv_files(self):
        candidates = [
            self.path("data/processed/passed_sessions.csv"),
            self.path("data/processed/rejected_sessions.csv"),
            self.path("data/processed/reject_sessions.csv"),
            self.path("data/processed/session_log.csv"),
            self.path("data/processed/reject_log.csv"),
            self.path("data/processed/hdf5_errors.csv"),
        ]
        return [p for p in candidates if p.is_file()]

    def check_csv_logs(self):
        files = self._csv_files()
        if not files:
            self.add("Session CSV logs", "NOT_CHECKED",
                     "No recognized session CSV logs found under data/processed/.")
            return
        evidence = {}
        problems = []
        for path in files:
            try:
                with path.open("r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    header = reader.fieldnames or []
                    rows = list(reader)
                evidence[str(path.relative_to(self.root))] = {
                    "rows": len(rows), "columns": header
                }
                if not header:
                    problems.append(f"{path.name}: empty/missing header")
                if any(len(row) != len(header) for row in rows):
                    problems.append(f"{path.name}: inconsistent row width")
            except Exception as exc:
                problems.append(f"{path.name}: {type(exc).__name__}: {exc}")
        if problems:
            self.add("Session CSV logs", "FAIL",
                     "CSV log structure problems detected.", problems)
        else:
            self.add("Session CSV logs", "PASS",
                     "Recognized CSV logs are readable and have consistent row widths.",
                     evidence)

        pass_file = self.path("data/processed/passed_sessions.csv")
        if pass_file.is_file():
            try:
                with pass_file.open("r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fields = reader.fieldnames or []
                normalized = {x.strip().lower() for x in fields}
                has_user = any(x in normalized for x in ("user_id", "userid", "user"))
                has_session = any(x in normalized for x in ("session_id", "sessionid", "session"))
                if has_user and has_session:
                    keys = []
                    for row in rows:
                        low = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                        user = next((low[k] for k in ("user_id", "userid", "user") if k in low), "")
                        session = next((low[k] for k in ("session_id", "sessionid", "session") if k in low), "")
                        keys.append((user, session))
                    duplicates = len(keys) - len(set(keys))
                    self.add("PASS session key integrity",
                             "PASS" if duplicates == 0 else "FAIL",
                             f"{len(rows)} rows; duplicate (user, session) keys: {duplicates}.",
                             {"columns": fields, "duplicate_keys": duplicates})
                else:
                    self.add("PASS session key integrity", "NOT_CHECKED",
                             "Could not identify user_id and session_id columns.",
                             fields)
            except Exception as exc:
                self.add("PASS session key integrity", "FAIL",
                         f"Could not parse passed_sessions.csv: {exc}")

        reject_candidates = [
            self.path("data/processed/rejected_sessions.csv"),
            self.path("data/processed/reject_sessions.csv"),
            self.path("data/processed/reject_log.csv"),
        ]
        if not any(p.is_file() for p in reject_candidates):
            self.add("Rejected-session accounting", "WARN",
                     "No recognized reject log found. PASS-only CSV cannot establish "
                     "how many sessions were rejected or why.")

    def check_hdf5(self):
        candidates = [
            self.path("data/processed/head_pose.h5"),
            self.path("data/processed/head_pose_v2.h5"),
        ]
        h5_files = [p for p in candidates if p.is_file()]
        if not h5_files:
            self.add("HDF5 artifact", "NOT_CHECKED",
                     "No recognized HDF5 output found.")
            return
        try:
            import h5py
            import numpy as np
        except Exception as exc:
            self.add("HDF5 artifact", "NOT_CHECKED",
                     f"h5py/numpy unavailable: {exc}")
            return

        for path in h5_files:
            evidence = {"size_bytes": path.stat().st_size, "datasets": []}
            failures, warnings = [], []
            try:
                with h5py.File(path, "r") as h5:
                    def visit(name, obj):
                        if not isinstance(obj, h5py.Dataset):
                            return
                        item = {
                            "path": name,
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                        }
                        evidence["datasets"].append(item)
                        if obj.size == 0:
                            warnings.append(f"{name}: empty dataset")
                            return
                        if obj.dtype.kind in "fiu":
                            try:
                                arr = obj[()]
                                if not np.isfinite(arr).all():
                                    failures.append(f"{name}: contains NaN/Inf")
                                low = name.lower()
                                if "quat" in low and arr.ndim >= 1 and arr.shape[-1] == 4:
                                    norms = np.linalg.norm(arr, axis=-1)
                                    bad = int(np.count_nonzero(~np.isfinite(norms) | (np.abs(norms - 1.0) > 1e-3)))
                                    if bad:
                                        failures.append(f"{name}: {bad} quaternion(s) not unit norm (tol=1e-3)")
                                if ("timestamp" in low or low.endswith("/time") or low == "time") and arr.ndim == 1 and arr.size > 1:
                                    delta = np.diff(arr.astype(np.float64))
                                    nonmono = int(np.count_nonzero(delta <= 0))
                                    if nonmono:
                                        failures.append(f"{name}: {nonmono} non-increasing timestamp intervals")
                            except Exception as exc:
                                warnings.append(f"{name}: could not inspect values ({exc})")
                    h5.visititems(visit)
                if not evidence["datasets"]:
                    failures.append("HDF5 contains no datasets")
                status = "FAIL" if failures else ("WARN" if warnings else "PASS")
                detail = f"{path.name}: {len(evidence['datasets'])} datasets; "
                detail += f"{len(failures)} failure(s), {len(warnings)} warning(s)."
                self.add(f"HDF5 structural/data checks: {path.name}", status,
                         detail, {"evidence": evidence, "failures": failures, "warnings": warnings})
            except Exception as exc:
                self.add(f"HDF5 structural/data checks: {path.name}", "FAIL",
                         f"Cannot open/read HDF5: {type(exc).__name__}: {exc}")

        if len(h5_files) > 1:
            self.add("Canonical HDF5 selection", "WARN",
                     "Both head_pose.h5 and head_pose_v2.h5 exist. Select one canonical "
                     "training artifact after comparing schemas and contents.",
                     [p.name for p in h5_files])

    def check_storage(self):
        managed = [
            self.path("data"),
            self.path("checkpoints"),
            self.path("outputs"),
            self.path("reports"),
            self.path("experiments"),
        ]
        seen: set[Path] = set()
        total = 0
        breakdown = {}
        for base in managed:
            if not base.exists():
                continue
            subtotal = 0
            paths = [base] if base.is_file() else base.rglob("*")
            for item in paths:
                try:
                    resolved = item.resolve()
                    if resolved in seen or not item.is_file():
                        continue
                    seen.add(resolved)
                    subtotal += item.stat().st_size
                except OSError:
                    continue
            breakdown[str(base.relative_to(self.root))] = subtotal
            total += subtotal

        free = shutil.disk_usage(self.root).free
        evidence = {
            "managed_bytes": total,
            "managed_gib": round(total / 1024**3, 3),
            "limit_gib": round(self.max_bytes / 1024**3, 3),
            "free_bytes": free,
            "free_gib": round(free / 1024**3, 3),
            "reserve_gib": round(self.reserve_bytes / 1024**3, 3),
            "breakdown_bytes": breakdown,
        }
        if total > self.max_bytes:
            self.add("Pipeline storage cap", "FAIL",
                     "Pipeline-managed files exceed the configured storage cap.", evidence)
        else:
            self.add("Pipeline storage cap", "PASS",
                     "Tracked pipeline-managed files are within the configured cap.", evidence)
        if free < self.reserve_bytes:
            self.add("Free disk reserve", "FAIL",
                     "Free disk space is below the configured safety reserve.", evidence)
        else:
            self.add("Free disk reserve", "PASS",
                     "Free disk space meets the configured safety reserve.", evidence)

    def check_source_safety_markers(self):
        checks = {
            "src/prep/resampling.py": [
                ("searchsorted", "Efficient nearest-timestamp lookup marker"),
                ("slerp", "SLERP interpolation marker"),
            ],
            "src/prep/session.py": [
                ("no_valid_segments", "Empty resampling result handling"),
                ("n_windows", "Window count handling"),
            ],
            "src/prep/windows.py": [
                ("context", "Context-window logic marker"),
            ],
        }
        for rel, markers in checks.items():
            path = self.path(rel)
            if not path.is_file():
                self.add(f"Source markers: {rel}", "NOT_CHECKED", "File missing.")
                continue
            content = path.read_text(errors="replace").lower()
            missing = [label for token, label in markers if token not in content]
            if missing:
                self.add(f"Source markers: {rel}", "WARN",
                         "Expected implementation markers not found; inspect manually.",
                         missing)
            else:
                self.add(f"Source markers: {rel}", "WARN",
                         "Markers found, but source-text checks do not prove correctness.",
                         [label for _, label in markers])

    def run(self):
        self.check_required_files()
        self.check_compile()
        self.check_imports()
        self.check_config()
        self.check_csv_logs()
        self.check_hdf5()
        self.check_storage()
        self.check_source_safety_markers()
        return self.results


def write_reports(root: Path, results: list[dict[str, Any]]):
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = report_dir / f"pipeline_audit_{stamp}.json"
    md_path = report_dir / f"pipeline_audit_{stamp}.md"

    counts = Counter(item["status"] for item in results)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": dict(counts),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    lines = [
        "# ABC preprocessing pipeline audit",
        "",
        f"- Generated (UTC): {payload['generated_at_utc']}",
        f"- PASS: {counts['PASS']} | WARN: {counts['WARN']} | "
        f"FAIL: {counts['FAIL']} | NOT_CHECKED: {counts['NOT_CHECKED']}",
        "",
        "> PASS means directly verified by this script. WARN/NOT_CHECKED items are not "
        "proof of correctness. Review all FAIL items before using the dataset for training.",
        "",
        "| Status | Check | Result |",
        "|---|---|---|",
    ]
    for item in results:
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| **{item['status']}** | {item['check']} | {detail} |")
    lines += ["", "## Evidence", ""]
    for item in results:
        lines += [
            f"### {item['status']} — {item['check']}",
            "",
            item["detail"],
            "",
        ]
        if item.get("evidence") is not None:
            lines += ["```json",
                      json.dumps(item["evidence"], indent=2, ensure_ascii=False, default=str),
                      "```", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--max-gb", type=float, default=10.0,
                        help="Maximum pipeline-managed storage in GiB (default: 10)")
    parser.add_argument("--reserve-free-gb", type=float, default=5.0,
                        help="Minimum free disk reserve in GiB (default: 5)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Repository root does not exist: {root}")

    audit = Audit(root, args.max_gb, args.reserve_free_gb)
    results = audit.run()
    md_path, json_path, counts = write_reports(root, results)

    print("\n=== ABC PIPELINE AUDIT ===")
    for item in results:
        print(f"[{item['status']:<11}] {item['check']}: {item['detail']}")
    print("\n=== SUMMARY ===")
    print(" | ".join(f"{key}: {counts[key]}" for key in STATUSES))
    print(f"Markdown report: {md_path}")
    print(f"JSON report:     {json_path}")

    if counts["FAIL"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
