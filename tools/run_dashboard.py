"""
Streamlit dashboard for browsing per-run training metrics and ladder results.

Layout:
  - Sidebar lists timestamped run directories under `runs/` and ladder
    backfills under `eval_results/`. Multi-select to overlay.
  - Main panel: line charts (one per metric) with per-series toggles and
    auto-refresh.

Conventions this dashboard expects:
  - Runs:        runs/<YYYYMMDD-HHMMSS>__<experiment-name>/metrics.csv
  - Backfills:   eval_results/<checkpoint-name>/ladder.csv
  - All CSVs are append-only line writes (atomic), so live runs update
    incrementally as the trainer flushes.

Launch from the repository root:

  streamlit run tools/run_dashboard.py

The browser tab stays open; new runs appear automatically as their dirs
materialize.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


RUNS_DIR = os.environ.get("BLEF_RUNS_DIR", "runs")
EVAL_DIR = os.environ.get("BLEF_EVAL_DIR", "eval_results")
RUN_DIRNAME_RE = re.compile(r"^(?P<ts>\d{8}-\d{6})__(?P<name>.+)$")


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------

def _list_runs(root: str) -> List[Dict]:
    """Return list of {dir, timestamp, name, csv_path} sorted newest-first."""
    if not os.path.isdir(root):
        return []
    out = []
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        m = RUN_DIRNAME_RE.match(entry)
        if not m:
            # Tolerate non-conforming dirs as "no timestamp" entries.
            csv_path = os.path.join(full, "metrics.csv")
            if os.path.exists(csv_path):
                out.append({
                    "dir": full,
                    "timestamp": "",
                    "name": entry,
                    "csv_path": csv_path,
                })
            continue
        csv_path = os.path.join(full, "metrics.csv")
        if not os.path.exists(csv_path):
            # Run dir created but no metrics yet; show it anyway.
            csv_path = ""
        out.append({
            "dir": full,
            "timestamp": m.group("ts"),
            "name": m.group("name"),
            "csv_path": csv_path,
        })
    out.sort(key=lambda r: r["timestamp"], reverse=True)
    return out


def _list_eval_backfills(root: str) -> List[Dict]:
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root), reverse=True):
        full = os.path.join(root, entry)
        ladder = os.path.join(full, "ladder.csv")
        if os.path.isdir(full) and os.path.exists(ladder):
            out.append({"name": entry, "csv_path": ladder, "dir": full})
    return out


@st.cache_data(ttl=5)
def _read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        # Tolerate concurrent appends: read what's there.
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Blef NFSP Dashboard", layout="wide")
    st.title("Blef NFSP — runs & eval ladder")

    # Auto-refresh button. Streamlit reruns on widget interaction; this lets
    # the user pull fresh CSVs without refreshing the whole page.
    refresh_secs = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 10)
    if refresh_secs > 0:
        st.sidebar.markdown(
            f"<small>Auto-refreshing every {refresh_secs}s. Disable: drag to 0.</small>",
            unsafe_allow_html=True,
        )

    runs = _list_runs(RUNS_DIR)
    backfills = _list_eval_backfills(EVAL_DIR)

    st.sidebar.header("Runs")
    if not runs:
        st.sidebar.write(f"No runs found under `{RUNS_DIR}/`.")
    selected_run_names = st.sidebar.multiselect(
        "Overlay runs",
        options=[r["name"] + (f" ({r['timestamp']})" if r["timestamp"] else "") for r in runs],
        default=[],
    )
    selected_runs = [
        r for r, label in zip(runs, [r["name"] + (f" ({r['timestamp']})" if r["timestamp"] else "") for r in runs])
        if label in selected_run_names
    ]

    st.sidebar.header("Eval backfills")
    selected_backfills = st.sidebar.multiselect(
        "Compare ladder backfills",
        options=[b["name"] for b in backfills],
        default=[],
    )

    # ----- Run metrics -----
    if selected_runs:
        st.subheader("Training metrics")
        frames = []
        for r in selected_runs:
            df = _read_csv_safe(r["csv_path"])
            if df is None or df.empty:
                continue
            df = df.copy()
            df["__run"] = r["name"]
            frames.append(df)
        if not frames:
            st.info("Selected runs have no metrics CSV yet (or the file is empty).")
        else:
            all_df = pd.concat(frames, ignore_index=True)
            x_col = "step" if "step" in all_df.columns else None
            metric_cols = [
                c for c in all_df.columns
                if c not in {"__run", "step"}
                and pd.api.types.is_numeric_dtype(all_df[c])
            ]
            picked = st.multiselect(
                "Metrics to chart",
                options=metric_cols,
                default=[c for c in ("avg_reward", "win_rate", "q_loss", "sl_loss", "policy_entropy") if c in metric_cols][:5],
            )
            ma_window = st.slider(
                "Moving-average window (samples)",
                min_value=1, max_value=200, value=20, step=1,
                help="1 = raw only. >1 overlays a rolling-mean trend line on each metric.",
            )
            show_raw = st.checkbox("Show raw alongside moving average", value=True)
            for col in picked:
                if x_col:
                    raw = (
                        all_df.pivot_table(index=x_col, columns="__run", values=col, aggfunc="last")
                              .sort_index()
                    )
                else:
                    raw = all_df[[col, "__run"]].pivot_table(columns="__run", values=col, aggfunc="last")
                st.markdown(f"**{col}**")
                if ma_window > 1:
                    ma = raw.rolling(window=ma_window, min_periods=1).mean()
                    ma.columns = [f"{c} (MA{ma_window})" for c in ma.columns]
                    if show_raw:
                        chart = pd.concat([raw, ma], axis=1)
                    else:
                        chart = ma
                else:
                    chart = raw
                st.line_chart(chart, use_container_width=True)

    # ----- Backfilled ladder -----
    if selected_backfills:
        st.subheader("Eval ladder (head-to-head winrate vs opponent)")
        frames = []
        chosen = [b for b in backfills if b["name"] in selected_backfills]
        for b in chosen:
            df = _read_csv_safe(b["csv_path"])
            if df is None or df.empty:
                continue
            df = df.copy()
            df["__checkpoint"] = b["name"]
            frames.append(df)
        if frames:
            ladder = pd.concat(frames, ignore_index=True)
            st.dataframe(ladder, use_container_width=True)
            if {"opponent", "winrate", "__checkpoint"}.issubset(ladder.columns):
                pivot = ladder.pivot_table(
                    index="__checkpoint",
                    columns="opponent",
                    values="winrate",
                    aggfunc="mean",
                ).sort_index()
                st.bar_chart(pivot)
        else:
            st.info("Selected backfills have no ladder.csv yet.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<small>Runs dir: `{RUNS_DIR}` · Eval dir: `{EVAL_DIR}`</small>", unsafe_allow_html=True)

    if refresh_secs > 0:
        time.sleep(refresh_secs)
        st.rerun()


if __name__ == "__main__":
    main()
