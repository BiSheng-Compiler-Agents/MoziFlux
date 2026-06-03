"""
gen_report.py — Aggregate performance plots for KernelBench_Triton optimizations

Scans every directory in datasets/KernelBench_Triton/ for a results.txt file
(produced by profile_kernels.py via @triton.testing.perf_report), extracts
the perf tables, and writes:

  1. One PNG per kernel directory  — runtime vs shape, one curve per method
  2. One aggregate PNG             — runtime distribution per method, pooled
                                     across all (source_dir, shape) data points
  3. One aggregate geomean PNG     — geomean speedup of Optimized vs Baseline
                                     vs Reference per kernel directory
  4. A CSV summary at reports/all_runtimes.csv

The kernel identifier everywhere is the DIRECTORY name (source_dir), not
the section name parsed out of results.txt — section names like
'matmul_benchmark' or 'relu_perf' are arbitrary per-file and unreliable
for cross-kernel grouping. The parser still reads the section header to
locate the perf table, but its text is not used as an identifier.

Usage:
    python gen_report.py                              # process every kernel dir
    python gen_report.py --source-dir l1_19_ReLU      # process a single dir
    python gen_report.py --output-dir reports/        # custom output dir
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "datasets" / "KernelBench_Triton"
OUTPUT_DIR  = ROOT / "reports"

# ── Method normalization ──────────────────────────────────────────────────────
# results.txt files use different names for the same logical reference impl.
# Normalize so the aggregate plot groups them together.
METHOD_ALIASES: dict[str, str] = {
    "PyTorch / ACL":           "Reference (PyTorch/ACL)",
    "Baseline Triton1":         "Baseline Triton1",
    "Baseline Triton2":         "Baseline Triton2",
    "Optimized Triton":        "Optimized Triton",
}
# Fixed color per logical method so colors are consistent across all plots.
METHOD_COLORS: dict[str, str] = {
    "Reference (PyTorch/ACL)": "#1f77b4",   # blue
    "Baseline Triton1":         "#ff7f0e",   # orange
    "Baseline Triton2":         "#ff320e",   # orange
    "Optimized Triton":        "#2ca02c",   # green
}
# Consistent line styles per method.
METHOD_STYLES: dict[str, str] = {
    "Reference (PyTorch/ACL)": "-",
    "Baseline Triton1":         "--",
    "Baseline Triton2":         "--",
    "Optimized Triton":        "-",
}
METHOD_MARKERS: dict[str, str] = {
    "Reference (PyTorch/ACL)": "o",
    "Baseline Triton1":         "s",
    "Baseline Triton2":         "s",
    "Optimized Triton":        "^",
}


# ── Parser ────────────────────────────────────────────────────────────────────
def _find_runs(line: str) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for every run of non-space chars."""
    runs = []
    in_run = False
    start = 0
    for i, c in enumerate(line):
        if c != " " and not in_run:
            start = i
            in_run = True
        elif c == " " and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(line)))
    return runs


def _is_section_header(stripped: str) -> bool:
    """A section header is a short single-token line ending with ':'.
    Rejects multi-word lines like 'Correctness check (atol=...):'."""
    if not stripped.endswith(":"):
        return False
    return " " not in stripped[:-1]


def _parse_header_columns(header_line: str, n_data_columns: int) -> list[str] | None:
    """Parse the column-name list from a pandas DataFrame.to_string() header.
    Merges adjacent non-space runs separated by exactly 1 character so
    multi-word names like 'Baseline Triton' and 'PyTorch / ACL' stay whole.
    Returns the first n_data_columns names."""
    runs = _find_runs(header_line)
    if not runs:
        return None
    merged: list[tuple[int, int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] == 1:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    names = [header_line[s:e].strip() for s, e in merged]
    if len(names) < n_data_columns:
        return None
    return names[:n_data_columns]


def _parse_data_row(data_line: str, n_columns: int) -> list[str] | None:
    """Parse a data row by finding the integer pandas index (first token)
    and the right-aligned values that follow. Returns [shape, m1, m2, ...]."""
    stripped = data_line.strip()
    if not stripped or not stripped.split()[0].isdigit():
        return None
    runs = _find_runs(data_line)
    if len(runs) < 1 + n_columns:
        return None
    return [data_line[r[0]:r[1]].strip() for r in runs[1:1 + n_columns]]


def _extract_pandas_blocks(text: str) -> list[tuple[str, pd.DataFrame]]:
    """Walk the file, find each perf section, return (section_name, DataFrame).
    Skips correctness sections and stray non-table lines."""
    out: list[tuple[str, pd.DataFrame]] = []
    section: str | None = None
    header_line: str | None = None
    n_columns: int | None = None
    rows: list[list[str]] = []

    def flush():
        if not (section and header_line and n_columns and rows):
            return
        names = _parse_header_columns(header_line, n_columns)
        if names is None:
            return
        df = pd.DataFrame(rows, columns=names)
        shape_col = names[0]
        df = df.rename(columns={shape_col: "shape"})
        for c in df.columns:
            if c == "shape":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(how="all", subset=[c for c in df.columns if c != "shape"])
        if not df.empty:
            out.append((section, df))

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if _is_section_header(stripped):
            flush()
            section = stripped[:-1]
            header_line = None
            n_columns = None
            rows = []
            continue
        if (
            stripped.startswith("Correctness check")
            or "[PASS]" in stripped
            or "maxΔ" in stripped
            or "maxDelta" in stripped
            or stripped == "All PASS"
        ):
            continue
        if header_line is None:
            if stripped.split()[0] in ("label", "N"):
                header_line = line
            continue
        if stripped.split()[0].isdigit():
            data_runs = _find_runs(line)
            if n_columns is None:
                n_columns = len(data_runs) - 1
            values = _parse_data_row(line, n_columns)
            if values:
                rows.append(values)
    flush()
    return out


def parse_results_txt(path: Path) -> list[dict]:
    """Parse a results.txt and return a list of perf records
    {source_dir, shape, method, runtime}. The kernel identifier is the
    directory name (source_dir), NOT the section name parsed out of
    results.txt — section names are arbitrary per-file and unreliable
    for cross-kernel grouping."""
    text = path.read_text()
    source_dir = path.parent.name
    records: list[dict] = []
    for _section, df in _extract_pandas_blocks(text):
        for method in df.columns:
            if method == "shape":
                continue
            for _, row in df.iterrows():
                rt = row[method]
                if pd.isna(rt):
                    continue
                records.append({
                    "source_dir": source_dir,
                    "shape":      str(row["shape"]),
                    "method":     METHOD_ALIASES.get(str(method), str(method)),
                    "runtime":    float(rt),
                })
    return records


def scan_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Find every results.txt under dataset_dir and return a DataFrame with
    columns: source_dir, shape, method, runtime. The kernel is identified
    by directory name only."""
    all_records: list[dict] = []
    for path in sorted(dataset_dir.glob("*/results.txt")):
        all_records.extend(parse_results_txt(path))
    return pd.DataFrame(
        all_records,
        columns=["source_dir", "shape", "method", "runtime"],
    )


# ── Shape size heuristic (for x-axis ordering on per-kernel plots) ───────────
_SHAPE_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _shape_sort_key(label: str) -> tuple[int, float, str]:
    """Return a sort key for a shape label. Numeric labels sort numerically;
    everything else falls back to lexicographic order."""
    nums = _SHAPE_NUM_RE.findall(label)
    if nums:
        try:
            return (0, float(nums[0]), label)
        except ValueError:
            pass
    return (1, 0.0, label)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _method_order(methods: list[str]) -> list[str]:
    """Stable order: Reference, Baseline, Optimized, then anything else."""
    priority = ["Reference (PyTorch/ACL)", "Baseline Triton", "Optimized Triton"]
    seen = set()
    out = [m for m in priority if m in methods and not (m in seen or seen.add(m))]
    out.extend(m for m in sorted(methods) if m not in seen)
    return out


def plot_per_kernel(
    df: pd.DataFrame,
    source_dir: str,
    out_path: Path,
) -> None:
    """Line plot: x = shape (sorted by size), y = runtime (log ms),
    one curve per method with a fixed color. Titled by the kernel
    directory name (source_dir)."""
    sub = df[df["source_dir"] == source_dir].copy()
    if sub.empty:
        return
    sub["__sort"] = sub["shape"].map(_shape_sort_key)
    sub = sub.sort_values("__sort")

    methods = _method_order(sub["method"].unique().tolist())
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        msub = sub[sub["method"] == m]
        ax.plot(
            msub["shape"], msub["runtime"],
            color=METHOD_COLORS.get(m, "#444444"),
            linestyle=METHOD_STYLES.get(m, "-"),
            marker=METHOD_MARKERS.get(m, "o"),
            markersize=7,
            linewidth=2,
            label=m,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Shape")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title(source_dir)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    # Rotate x labels for readability when there are many shapes
    n_shapes = sub["shape"].nunique()
    if n_shapes > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_aggregate_box(df: pd.DataFrame, out_path: Path) -> None:
    """Box plot: x = method, y = runtime (log), pooling every (kernel, shape)
    data point. Shows the overall runtime distribution per method."""
    methods = _method_order(df["method"].unique().tolist())
    data = [df[df["method"] == m]["runtime"].values for m in methods]
    colors = [METHOD_COLORS.get(m, "#444444") for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        data, tick_labels=methods, patch_artist=True, showfliers=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title(
        f"Aggregate runtime distribution across {df['source_dir'].nunique()} kernel(s) "
        f"and {df['shape'].nunique()} shape(s)"
    )
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _compute_geomean_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(source_dir, shape) speedups and return geomean per source_dir.
    Returns DataFrame with columns: source_dir, speedup_vs_baseline1,
    speedup_vs_baseline2 (if available), speedup_vs_ref (if available)."""
    wide = df.pivot_table(
        index=["source_dir", "shape"], columns="method",
        values="runtime", aggfunc="first",
    ).reset_index()
    if "Baseline Triton1" not in wide.columns or "Optimized Triton" not in wide.columns:
        return pd.DataFrame()

    has_ref = "Reference (PyTorch/ACL)" in wide.columns
    has_b2 = "Baseline Triton2" in wide.columns

    wide["speedup_vs_baseline1"] = wide["Baseline Triton1"] / wide["Optimized Triton"]
    if has_b2:
        wide["speedup_vs_baseline2"] = wide["Baseline Triton2"] / wide["Optimized Triton"]
    if has_ref:
        wide["speedup_vs_ref"] = wide["Reference (PyTorch/ACL)"] / wide["Optimized Triton"]

    agg = {"speedup_vs_baseline1": ("speedup_vs_baseline1", lambda s: np.exp(np.log(s).mean()))}
    if has_b2:
        agg["speedup_vs_baseline2"] = ("speedup_vs_baseline2", lambda s: np.exp(np.log(s).mean()))
    if has_ref:
        agg["speedup_vs_ref"] = ("speedup_vs_ref", lambda s: np.exp(np.log(s).mean()))

    geomean = wide.groupby("source_dir").agg(**agg).reset_index().sort_values("source_dir")
    return geomean


def _plot_single_speedup_bar(
    geomean: pd.DataFrame,
    col: str,
    color: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Single-bar geomean speedup chart: one bar per kernel directory."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(geomean))
    ax.bar(x, geomean[col], width=0.5, color=color, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(geomean["source_dir"], rotation=30, ha="right")
    ax.set_xlabel("Kernel directory")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_geomean_speedup_vs_baseline1(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: geomean speedup of Optimized Triton vs Baseline Triton1 per kernel."""
    geomean = _compute_geomean_speedups(df)
    if geomean.empty:
        return
    _plot_single_speedup_bar(
        geomean,
        col="speedup_vs_baseline1",
        color="#ff7f0e",
        title="Geometric-mean speedup: Optimized Triton vs Baseline Triton1",
        ylabel="Geometric mean speedup (×)\n>1 means Optimized is faster",
        out_path=out_path,
    )


def plot_geomean_speedup_vs_baseline2(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: geomean speedup of Optimized Triton vs Baseline Triton2 per kernel."""
    geomean = _compute_geomean_speedups(df)
    if geomean.empty or "speedup_vs_baseline2" not in geomean.columns:
        return
    _plot_single_speedup_bar(
        geomean,
        col="speedup_vs_baseline2",
        color="#ff320e",
        title="Geometric-mean speedup: Optimized Triton vs Baseline Triton2",
        ylabel="Geometric mean speedup (×)\n>1 means Optimized is faster",
        out_path=out_path,
    )


def plot_geomean_speedup_vs_ref(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: geomean speedup of Optimized Triton vs Reference (PyTorch/ACL) per kernel."""
    geomean = _compute_geomean_speedups(df)
    if geomean.empty or "speedup_vs_ref" not in geomean.columns:
        return
    _plot_single_speedup_bar(
        geomean,
        col="speedup_vs_ref",
        color="#1f77b4",
        title="Geometric-mean speedup: Optimized Triton vs Reference (PyTorch/ACL)",
        ylabel="Geometric mean speedup (×)\n>1 means Optimized is faster",
        out_path=out_path,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-kernel and aggregate performance plots "
                    "from datasets/KernelBench_Triton/*/results.txt"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help="Directory containing kernel subdirs with results.txt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Where to write PNGs and the CSV summary",
    )
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help="Process a single kernel directory (e.g. l1_19_ReLU) instead of all",
    )
    args = parser.parse_args()

    per_kernel_dir = args.output_dir / "per_kernel"
    per_kernel_dir.mkdir(parents=True, exist_ok=True)

    if args.source_dir:
        pattern = f"{args.source_dir}/results.txt"
        files = sorted(args.dataset_dir.glob(pattern))
        if not files:
            raise SystemExit(f"No results.txt found at {args.dataset_dir / pattern}")
    else:
        files = sorted(args.dataset_dir.glob("*/results.txt"))

    if not files:
        raise SystemExit(f"No results.txt files under {args.dataset_dir}")

    print(f"Scanning {len(files)} results.txt file(s) under {args.dataset_dir} ...")
    all_records: list[dict] = []
    for p in files:
        recs = parse_results_txt(p)
        all_records.extend(recs)
        print(f"  {p.parent.name:50s}  recs={len(recs)}")

    df = pd.DataFrame(
        all_records, columns=["source_dir", "shape", "method", "runtime"]
    )
    if df.empty:
        raise SystemExit("No perf records parsed — nothing to plot.")

    # CSV dump (useful for ad-hoc analysis)
    csv_path = args.output_dir / "all_runtimes.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    # Per-kernel plots: one PNG per kernel directory
    grouped = df.groupby("source_dir", sort=False)
    for source_dir, _sub in grouped:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_dir)
        out = per_kernel_dir / f"{safe}.png"
        plot_per_kernel(df, source_dir, out)
        print(f"  per-kernel: {out.relative_to(ROOT)}")

    # Aggregate plots
    agg_box = args.output_dir / "aggregate_runtime_boxplot.png"
    plot_aggregate_box(df, agg_box)
    print(f"  aggregate:  {agg_box.relative_to(ROOT)}")

    agg_speedup_b1 = args.output_dir / "aggregate_geomean_speedup_vs_baseline1.png"
    plot_geomean_speedup_vs_baseline1(df, agg_speedup_b1)
    print(f"  aggregate:  {agg_speedup_b1.relative_to(ROOT)}")

    agg_speedup_b2 = args.output_dir / "aggregate_geomean_speedup_vs_baseline2.png"
    plot_geomean_speedup_vs_baseline2(df, agg_speedup_b2)
    print(f"  aggregate:  {agg_speedup_b2.relative_to(ROOT)}")

    agg_speedup_ref = args.output_dir / "aggregate_geomean_speedup_vs_ref.png"
    plot_geomean_speedup_vs_ref(df, agg_speedup_ref)
    print(f"  aggregate:  {agg_speedup_ref.relative_to(ROOT)}")

    print(f"\nDone. {len(grouped)} per-kernel plot(s) + 4 aggregate plot(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
