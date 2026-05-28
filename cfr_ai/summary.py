"""Unified training + LBR summary CSV.

One row per setup at `cfr_ai/outputs/summary_of_all_runs.csv`. Training and
LBR own disjoint column sets:

  Training cols  (written by `cfr_ai/training.py` on save):
    Setup, Finished, Iterations, Penalty, Min bet, Pruning threshold,
    Minimum regret, Duration, Nodes touched, Explored infosets,
    Non-checking infosets, RAM (MB), P0 value, P1 value, Version

  LBR cols       (written by `cfr_ai/lbr.py` on `--update-summary`):
    LBR-<d> expl, LBR-<d> duration  (one pair per depth d evaluated)
    Sampling                        (the sampling params used)

When training writes a row, it blanks that row's LBR cols (the trained
policy changed, so old exploitability is stale). When LBR writes a row, it
creates the row with empty training cols if one doesn't exist yet.

The LBR depth column pairs grow rightward as more depths get computed; the
helpers preserve existing depth columns across writes.

The old `lbr_summary.csv` is superseded by this file. The training-analytics
rebuild path (`cfr_ai/analysis/training_analytics.py`) is the one-shot
"regenerate everything from outputs/*/metadata.csv" tool, useful after a
migration or to fix a corrupted file.
"""

import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


SUMMARY_PATH = os.path.join("cfr_ai", "outputs", "summary_of_all_runs.csv")


TRAINING_COLS: List[str] = [
    "Finished",
    "Iterations",
    "Penalty",
    "Min bet",
    "Pruning threshold",
    "Minimum regret",
    "Duration",
    "Nodes touched",
    "Explored infosets",
    "Non-checking infosets",
    "RAM (MB)",
    "P0 value",
    "P1 value",
    "Version",
]

# Non-depth LBR cols (shared across all depths in a row).
LBR_FIXED_COLS: List[str] = ["Sampling"]

# Detects LBR depth columns like "LBR-1 expl", "LBR-2 duration", "LBR-inf expl".
_LBR_DEPTH_RE = re.compile(r"^LBR-(\d+|inf) (expl|duration)$")


def _depth_sort_key(label: str):
    """Sort key for LBR depth labels: digit strings by int value, 'inf' last."""
    return float("inf") if label == "inf" else int(label)


# ---------------------------------------------------------------------------
# Setup-key helpers
# ---------------------------------------------------------------------------

def setup_key(hand_sizes) -> str:
    """Canonical row-key form: ``"3,7"`` (comma-separated, sorted)."""
    xs = sorted(int(x) for x in hand_sizes)
    return ",".join(str(x) for x in xs)


def _setup_sort_key(key: str):
    try:
        xs = [int(x) for x in key.split(",")]
        return (sum(xs), xs)
    except Exception:
        return (10**9, [])


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _depth_cols(depth_label: str) -> Tuple[str, str]:
    return f"LBR-{depth_label} expl", f"LBR-{depth_label} duration"


def _depth_labels_in_rows(rows: Dict[str, Dict[str, str]]) -> List[str]:
    labels = set()
    for r in rows.values():
        for col in r.keys():
            m = _LBR_DEPTH_RE.match(col)
            if m:
                labels.add(m.group(1))
    return sorted(labels, key=_depth_sort_key)


def _build_fields(rows: Dict[str, Dict[str, str]],
                  extra_depth_labels: Optional[List[str]] = None) -> List[str]:
    """Construct the full ordered field list given the rows currently in the
    file plus any depth labels we're about to add. Training cols come first,
    then per-depth LBR cols, then the fixed LBR cols on the right."""
    labels = set(_depth_labels_in_rows(rows))
    if extra_depth_labels:
        labels.update(extra_depth_labels)
    depth_cols: List[str] = []
    for label in sorted(labels, key=_depth_sort_key):
        e, d = _depth_cols(label)
        depth_cols.extend([e, d])
    return ["Setup"] + TRAINING_COLS + depth_cols + LBR_FIXED_COLS


def _all_lbr_cols(fields: List[str]) -> List[str]:
    """LBR-related cols in `fields` (depth pairs + fixed LBR cols)."""
    return [c for c in fields
            if _LBR_DEPTH_RE.match(c) or c in LBR_FIXED_COLS]


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def read_summary(path: str = SUMMARY_PATH) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Returns ({setup_key: row_dict}, fieldnames) from the on-disk CSV.
    Returns ({}, []) if the file doesn't exist."""
    if not os.path.exists(path):
        return {}, []
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for r in reader:
            key = r.get("Setup")
            if not key:
                continue
            rows[key] = dict(r)
    return rows, fields


def write_summary(rows: Dict[str, Dict[str, str]],
                  path: str = SUMMARY_PATH,
                  extra_depth_labels: Optional[List[str]] = None) -> None:
    """Atomically write `rows` to `path` with the canonical field order."""
    fields = _build_fields(rows, extra_depth_labels=extra_depth_labels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows.keys(), key=_setup_sort_key):
            row = {k: rows[key].get(k, "") for k in fields}
            w.writerow(row)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public update API
# ---------------------------------------------------------------------------

def update_training_row(hand_sizes, training_data: Dict[str, object],
                        path: str = SUMMARY_PATH) -> None:
    """Replace this setup's training cols and blank its LBR cols.

    `training_data` should provide string-or-stringifiable values for keys in
    `TRAINING_COLS`. Missing keys are written as empty strings.
    """
    key = setup_key(hand_sizes)
    rows, _ = read_summary(path)
    row = rows.get(key, {"Setup": key})
    for col in TRAINING_COLS:
        row[col] = "" if training_data.get(col) is None else str(training_data[col])
    # Blank ALL existing LBR cols — new training invalidates exploitability.
    fields_so_far = _build_fields(rows)
    for col in _all_lbr_cols(fields_so_far):
        row[col] = ""
    rows[key] = row
    write_summary(rows, path=path)


def update_lbr_row(hand_sizes, depth_label,
                   expl_str: str, duration_str: str,
                   sampling_label: str,
                   path: str = SUMMARY_PATH) -> None:
    """Replace this setup's LBR-<depth_label> cols (and the shared Sampling
    cell) in the unified summary. Creates the row with empty training cols if
    it doesn't exist yet (LBR-on-archived-snapshot case). `depth_label` is a
    string like "1", "2", or "inf" (for exact best response)."""
    key = setup_key(hand_sizes)
    rows, _ = read_summary(path)
    row = rows.get(key, {"Setup": key})
    label = str(depth_label)
    e_col, d_col = _depth_cols(label)
    row[e_col] = expl_str
    row[d_col] = duration_str
    row["Sampling"] = sampling_label
    rows[key] = row
    write_summary(rows, path=path, extra_depth_labels=[label])


# ---------------------------------------------------------------------------
# Per-setup metadata.csv LBR block
# ---------------------------------------------------------------------------

# Matches both the new ("LBR Exploitability") and the legacy v0 ("Exploitability")
# block markers, so a snapshot archived with the old format gets replaced
# cleanly when the new LBR runs.
_LBR_SECTION_RE = re.compile(r"^---\s.*xploitability\b", re.IGNORECASE)
_LBR_DEPTH_LINE_RE = re.compile(r"^LBR-(\d+|inf) (expl|duration)$")


def update_metadata_lbr(setup_dir: str, depth_label: str,
                        expl_str: str, duration_str: str,
                        sampling_label: str) -> None:
    """Append/update the `--- LBR Exploitability ---` block at the end of
    `setup_dir/metadata.csv`. Preserves prior depths' rows in the block, and
    replaces the legacy v0 `--- Exploitability ---` block if present (it
    didn't carry depth-aware rows, so there's nothing to preserve from it).

    Silently no-ops if `metadata.csv` doesn't exist — the LBR-on-archived-
    snapshot or LBR-without-prior-training case is already handled by the
    unified summary file.
    """
    md_path = os.path.join(setup_dir, "metadata.csv")
    if not os.path.exists(md_path):
        return

    rows = _read_csv_rows(md_path)
    cutoff = len(rows)
    for i, row in enumerate(rows):
        if row and _LBR_SECTION_RE.match(row[0]):
            cutoff = i
            break

    head = rows[:cutoff]

    existing_depths: Dict[str, Tuple[str, str]] = {}
    existing_sampling = sampling_label
    for row in rows[cutoff + 1:]:
        if not row or len(row) < 2:
            continue
        k, v = row[0], row[1]
        if k == "Sampling":
            existing_sampling = v
            continue
        m = _LBR_DEPTH_LINE_RE.match(k)
        if not m:
            continue
        d = m.group(1)
        e, dur = existing_depths.get(d, ("", ""))
        if m.group(2) == "expl":
            e = v
        else:
            dur = v
        existing_depths[d] = (e, dur)

    existing_depths[depth_label] = (expl_str, duration_str)
    # The new sampling label always wins (we wrote the new run after all).
    existing_sampling = sampling_label

    with open(md_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in head:
            w.writerow(row)
        w.writerow(["--- LBR Exploitability ---", ""])
        w.writerow(["Sampling", existing_sampling])
        for d in sorted(existing_depths.keys(), key=_depth_sort_key):
            e, dur = existing_depths[d]
            w.writerow([f"LBR-{d} expl", e])
            w.writerow([f"LBR-{d} duration", dur])


# ---------------------------------------------------------------------------
# Bulk rebuild (used by analysis/training_analytics.py and the one-shot migrator)
# ---------------------------------------------------------------------------

def _read_csv_rows(path: str) -> List[List[str]]:
    """Read all CSV rows from `path`, falling back to latin-1 for legacy
    metadata files that embedded the `±` character via cp1252."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=encoding) as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"could not decode {path} as utf-8 or latin-1")


def parse_metadata_csv(path: str) -> Dict[str, str]:
    """Parse a per-setup `metadata.csv` (key/value rows) into a flat dict.
    Utility-log rows (`P0 Utility at Iter N`) and the `--- Utility Log ---`
    separator are kept verbatim so callers that want them can pick them out.
    """
    data: Dict[str, str] = {}
    for row in _read_csv_rows(path):
        if not row or len(row) < 2:
            continue
        k, v = row[0], row[1]
        if k == "k" and v == "v":  # header row, skip
            continue
        data[k.strip()] = v.strip()
    return data


def metadata_to_training_cols(meta: Dict[str, str]) -> Dict[str, str]:
    """Project a parsed metadata.csv onto the unified summary's training cols.
    Older metadata.csv files may not have every field — missing keys map to
    empty strings (rendered as blank cells)."""
    def _short_finished(s: str) -> str:
        # Old metadata used "YYYY-MM-DD, HH:MM:SS"; keep ISO date for the summary.
        if not s:
            return ""
        try:
            return datetime.strptime(s, "%Y-%m-%d, %H:%M:%S").strftime("%Y-%m-%d")
        except ValueError:
            return s
    return {
        "Finished": _short_finished(meta.get("Time finished", "")),
        "Iterations": meta.get("Iterations", ""),
        "Penalty": meta.get("Penalty", ""),
        "Min bet": meta.get("Minimum bet", ""),
        "Pruning threshold": meta.get("Pruning threshold", ""),
        "Minimum regret": meta.get("Minimum regret", ""),
        "Duration": meta.get("Training duration", ""),
        "Nodes touched": meta.get("Nodes touched", ""),
        "Explored infosets": meta.get("Explored infosets", ""),
        "Non-checking infosets": meta.get("Non-checking infosets", ""),
        "RAM (MB)": _round_ram(meta.get("RAM taken (MB)", "")),
        "P0 value": _fmt_value(meta.get("Player 1 game value", "")),
        "P1 value": _fmt_value(meta.get("Player 2 game value", "")),
        "Version": meta.get("Version code", ""),
    }


def _round_ram(s: str) -> str:
    try:
        return str(round(float(s)))
    except (TypeError, ValueError):
        return s or ""


def _fmt_value(s: str) -> str:
    try:
        return f"{float(s):.4f}"
    except (TypeError, ValueError):
        return s or ""


def rebuild_from_metadata(outputs_dir: str = os.path.join("cfr_ai", "outputs"),
                          preserve_lbr_from: Optional[str] = SUMMARY_PATH,
                          path: str = SUMMARY_PATH) -> int:
    """Walk `outputs_dir/*/metadata.csv`, project each to the training cols of
    the unified summary, then merge with any LBR cols from `preserve_lbr_from`
    (defaults to the current summary file). LBR cols are cleared whenever the
    `Finished` timestamp for that setup differs between the new training
    metadata and the existing summary — the same auto-invalidation that
    `update_training_row` applies on a single setup.

    Returns the number of training rows written.
    """
    # Step 1: existing LBR cols, keyed by setup, indexed by training timestamp.
    existing_rows: Dict[str, Dict[str, str]] = {}
    if preserve_lbr_from and os.path.exists(preserve_lbr_from):
        existing_rows, _ = read_summary(preserve_lbr_from)

    new_rows: Dict[str, Dict[str, str]] = {}
    extra_depth_labels: List[str] = []
    for entry in sorted(os.scandir(outputs_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        md_path = os.path.join(entry.path, "metadata.csv")
        if not os.path.exists(md_path):
            continue
        setup_name = entry.name.replace("_", ",")
        meta = parse_metadata_csv(md_path)
        row = {"Setup": setup_name, **metadata_to_training_cols(meta)}

        existing = existing_rows.get(setup_name)
        if existing is not None and existing.get("Finished") == row["Finished"]:
            for col, val in existing.items():
                if col in row:
                    continue  # training col, already set
                row[col] = val
                m = _LBR_DEPTH_RE.match(col)
                if m:
                    extra_depth_labels.append(m.group(1))
        new_rows[setup_name] = row

    write_summary(new_rows, path=path, extra_depth_labels=extra_depth_labels)
    return len(new_rows)
