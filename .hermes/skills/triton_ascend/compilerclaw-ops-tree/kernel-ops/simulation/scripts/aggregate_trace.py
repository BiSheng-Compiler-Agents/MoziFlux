#!/usr/bin/env python3
"""Aggregate a CANNSIM Chrome-trace JSON into a compact human/LLM-readable
text report (monospace tables).

Usage:
  python3 aggregate_trace.py <trace_core0.json> [--out summary.txt]
                                                [--top-instr N]
                                                [--top-events N]

Output is text, not JSON, because the consumer is an LLM. A few light
ranking-based annotations (← BOTTLENECK / ← CRITICAL) are included with
transparent rules so the reader can verify or ignore them.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import json

DEFAULT_TOP_INSTR = 12
DEFAULT_TOP_EVENTS = 6

# Annotation rule: an instruction is CRITICAL if its average single-event
# duration is at least this fraction of wall_cycles.
CRITICAL_AVG_FRAC = 0.25

# Factual descriptions of each CANNSIM pipeline. Shown inline in the pipeline
# table so the LLM has the hardware context without needing a separate glossary.
PIPELINE_ROLE = {
    "00_FC": "flow control / front-end",
    "01_SCALAR": "scalar pipeline",
    "02_SCALARLDST": "scalar load/store",
    "03_MTE1": "DMA / MTE1 (L1 -> L0A/L0B)",
    "04_MTE2": "DMA / MTE2 (GM -> L1/UB)",
    "05_VEC": "vector pipeline (compute + L0C -> UB moves)",
    "06_CUBE": "matrix / Cube pipeline",
    "07_MTE3": "DMA / MTE3 (UB -> GM or UB -> L1)",
    "10_PUSHQ": "queue push / dispatch",
    "11_RVECSU": "vector scalar/support unit",
    "12_RVECEX": "vector execute pipeline",
    "13_RVECLD": "vector load from UB",
    "14_RVECST": "vector store to UB",
    "15_FLOWCTRL": "flow control / sync",
    "16_FLOWCONTROL": "flow control / sync",
    "17_FIXP": "FIX pipe / L0C output path (L0C -> GM/L1)",
    "18_RVECLP": "vector loop/control pipeline",
    "19_Q": "queue / dispatch",
}


def merge_intervals(intervals):
    if not intervals:
        return 0, None, None
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total, intervals[0][0], cur_e


def strip_prefix(pipe_name: str) -> str:
    """02_SCALARLDST -> SCALARLDST"""
    return re.sub(r"^\d+_", "", pipe_name)


def fmt_table(headers, rows, aligns, annotations=None):
    """Render a fixed-width text table.

    aligns is a string of '<' / '>' per column (left/right).
    annotations is a list (same length as rows) of trailing strings appended
    after each row (e.g. '  ← BOTTLENECK'), or None for none.
    """
    cols = [list(map(str, col)) for col in zip(*([headers] + rows))] if rows else [[h] for h in headers]
    widths = [max(len(c) for c in col) for col in cols]
    annotations = annotations or [""] * len(rows)

    def fmt_row(values, trailing=""):
        parts = []
        for v, w, a in zip(values, widths, aligns):
            parts.append(f"{str(v):{a}{w}}")
        return "  ".join(parts) + trailing

    lines = [fmt_row(headers)]
    for r, ann in zip(rows, annotations):
        lines.append(fmt_row(r, ann))
    return "\n".join(lines)


def aggregate(trace_path: Path, top_instr: int, top_events: int, label: str = "") -> str:
    with open(trace_path) as f:
        events = json.load(f)

    pid_name = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_name[e["pid"]] = e["args"]["name"]

    x_events = [e for e in events if e.get("ph") == "X"]
    i_events = [e for e in events if e.get("ph") == "i"]
    if not x_events:
        return "(empty trace -- no duration events)"

    t_start = min(e["ts"] for e in x_events)
    t_end = max(e["ts"] + e.get("dur", 0) for e in x_events)
    wall_cycles = t_end - t_start

    # ---- pipeline aggregation ----
    intervals_per_pid = defaultdict(list)
    count_per_pid = defaultdict(int)
    lanes_per_pid = defaultdict(set)
    sum_dur_per_pid = defaultdict(int)
    for e in x_events:
        pid = e["pid"]
        dur = e.get("dur", 0)
        intervals_per_pid[pid].append((e["ts"], e["ts"] + dur))
        count_per_pid[pid] += 1
        lanes_per_pid[pid].add(e["tid"])
        sum_dur_per_pid[pid] += dur

    pipe_rows_full = []
    for pid in intervals_per_pid:
        busy, first, last = merge_intervals(intervals_per_pid[pid])
        pipe_rows_full.append({
            "name": pid_name.get(pid, str(pid)),
            "ops": count_per_pid[pid],
            "busy": busy,
            "lane_sum": sum_dur_per_pid[pid],
            "lanes": len(lanes_per_pid[pid]),
            "window": [first, last],
        })
    pipe_rows_full.sort(key=lambda r: -r["busy"])
    max_busy = pipe_rows_full[0]["busy"] if pipe_rows_full else 0

    pipe_headers = ["pipeline", "ops", "busy_cyc", "lane_sum", "lanes", "window"]
    pipe_table_rows = [[
        r["name"],
        r["ops"], r["busy"], r["lane_sum"], r["lanes"],
        f"[{r['window'][0]},{r['window'][1]}]",
    ] for r in pipe_rows_full]
    pipe_anns = [
        "  ← BOTTLENECK" if r["busy"] == max_busy else "" for r in pipe_rows_full
    ]

    # ---- instruction aggregation, per (name, pipeline) ----
    by_pair = defaultdict(lambda: {"count": 0, "total": 0})
    for e in x_events:
        key = (e["name"], pid_name.get(e["pid"], str(e["pid"])))
        s = by_pair[key]
        s["count"] += 1
        s["total"] += e.get("dur", 0)
    instr_full = [
        {"name": n, "pipe": p, "cnt": s["count"], "total": s["total"],
         "avg": s["total"] / s["count"]}
        for (n, p), s in by_pair.items()
    ]
    instr_full.sort(key=lambda r: -r["total"])

    instr_top = instr_full[:top_instr]
    max_total = instr_top[0]["total"] if instr_top else 0
    critical_avg = CRITICAL_AVG_FRAC * wall_cycles
    instr_headers = ["instruction", "pipe", "cnt", "total_cyc", "avg_cyc"]
    instr_rows = [[
        r["name"], strip_prefix(r["pipe"]), r["cnt"], r["total"], f"{r['avg']:.0f}",
    ] for r in instr_top]
    instr_anns = []
    for r in instr_top:
        crit = (r["total"] == max_total) or (r["avg"] >= critical_avg)
        instr_anns.append("  ← CRITICAL" if crit else "")

    # ---- worst single events, deduplicated by (name, pipeline) ----
    events_by_group = defaultdict(list)
    for e in x_events:
        events_by_group[(e["name"], pid_name.get(e["pid"], str(e["pid"])))].append(e)
    # rank groups by their max-duration event
    ranked_groups = sorted(
        events_by_group.items(),
        key=lambda kv: -max(e.get("dur", 0) for e in kv[1]),
    )[:top_events]
    worst_lines = []
    for (name, pipe), evs in ranked_groups:
        evs = sorted(evs, key=lambda e: -e.get("dur", 0))
        head, rest = evs[0], evs[1:]
        line = f"{name}@{strip_prefix(pipe)}  ts={head['ts']}  dur={head.get('dur', 0)}"
        if rest:
            avg_rest = sum(e.get("dur", 0) for e in rest) / len(rest)
            line += f"  (×{len(rest)} similar, avg ~{avg_rest:.0f})"
        worst_lines.append(line)

    # ---- control flow / instant event counts ----
    instant_counts = Counter(e["name"] for e in i_events).most_common()
    ctrl_line = "  ".join(f"{n}×{c}" for n, c in instant_counts)

    # ---- assemble ----
    out = []
    out.append(f"=== {label} Profile ===")
    out.append(f"wall_cycles: {wall_cycles}  |  x_events: {len(x_events)}  |  i_events: {len(i_events)}")
    out.append(f"time_window: [{t_start},{t_end}]")
    out.append("")
    out.append("--- Pipeline Utilization ---")
    out.append(fmt_table(pipe_headers, pipe_table_rows, "<>>>>>", pipe_anns))
    out.append("")
    out.append(f"--- Top Instructions by Cycle Cost (top {len(instr_top)} of {len(instr_full)}) ---")
    out.append(fmt_table(instr_headers, instr_rows, "<<>>>", instr_anns))
    out.append("")
    out.append(f"--- Worst Single Events (top {len(ranked_groups)} distinct, deduped) ---")
    out.extend(worst_lines)
    out.append("")
    out.append("--- Control Flow / Instant Events ---")
    out.append(ctrl_line if ctrl_line else "(none)")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path, help="trace_core0.json from CANNSIM")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <trace_dir>/trace_summary.txt)")
    ap.add_argument("--top-instr", type=int, default=DEFAULT_TOP_INSTR,
                    help=f"rows in instructions table (default {DEFAULT_TOP_INSTR})")
    ap.add_argument("--top-events", type=int, default=DEFAULT_TOP_EVENTS,
                    help=f"distinct groups in worst-events list (default {DEFAULT_TOP_EVENTS})")
    args = ap.parse_args()

    if not args.trace.exists():
        print(f"trace not found: {args.trace}", file=sys.stderr)
        sys.exit(1)

    report = aggregate(args.trace, args.top_instr, args.top_events, label=str(args.trace))
    out = args.out or args.trace.with_name("trace_summary.txt")
    with open(out, "w") as f:
        f.write(report)
    print(f"wrote {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
