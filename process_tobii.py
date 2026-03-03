from __future__ import annotations

import subprocess
import sys
import time
import zipfile
from pathlib import Path
import re
from typing import Iterable, Optional

import pandas as pd
import polars as pl


def get_base_dir() -> Path:
    """
    Prompt the user for the folder containing Tobii recordings.
    Repeats until a valid directory path is given.
    """
    while True:
        path_str = input("Enter the path to your Tobii recordings folder: ").strip()
        if not path_str:
            print("Please enter a path.")
            continue
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            print(f"Path does not exist: {path}")
            continue
        if not path.is_dir():
            print(f"Path is not a directory: {path}")
            continue
        return path


def find_candidate_data_dirs(root: Path) -> Iterable[Path]:
    """
    Yield directories that look like Tobii Glasses data dirs,
    i.e. they contain gazedata.gz, imudata.gz and scenevideo.mp4.
    """
    required_files = {"gazedata.gz", "imudata.gz", "scenevideo.mp4"}

    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        names = {p.name for p in path.iterdir() if p.is_file()}
        if required_files.issubset(names):
            yield path


def unpack_g3_archives(root: Path, out_root: Path) -> Iterable[Path]:
    """
    Find .g3 files under root and, for each one that is a valid zip archive,
    unpack it into a directory under out_root.

    Returns the paths of the unpacked directories.
    """
    all_g3 = list(root.rglob("*.g3"))
    for f in all_g3:
        if not zipfile.is_zipfile(f):
            print(f"Skipping non-zip .g3 file: {f}")
    g3_files = [f for f in all_g3 if zipfile.is_zipfile(f)]
    total = len(g3_files)
    if total == 0:
        return

    for i, g3_file in enumerate(g3_files, 1):
        target_dir = out_root / (g3_file.stem + "_raw")
        if target_dir.exists():
            print(f"[{i}/{total}] Already unpacked, skipping: {target_dir}")
            yield target_dir
            continue

        print(f"[{i}/{total}] Unpacking {g3_file.name} -> {target_dir.name} ...")
        start = time.time()
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(g3_file, "r") as zf:
            zf.extractall(target_dir)
        elapsed = time.time() - start
        print(f"       Done in {elapsed:.1f}s")
        yield target_dir


def run_tobii_munger_convert(data_dir: Path, out_parquet: Path) -> bool:
    """
    Call `python -m tobii_munger.convert` on data_dir to produce out_parquet.

    Returns True on success, False on failure.
    """
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tobii_munger.convert",
        str(data_dir),
        str(out_parquet),
    ]
    print("       Running tobii-munger (typically 2-8 min per recording)...")
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        print(f"       Converted in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"       tobii-munger failed for {data_dir}: {exc}")
        return False


def process_unified_parquet(parquet_path: Path) -> Optional[Path]:
    """
    Load a unified Parquet file and write out CSVs that are actually usable.

    Notes:
    - `tobii-munger` writes a "long-form" table with columns like:
      `timestamp`, `type`, `vals` (list)
    - A single timestamp can appear multiple times (one per `type`)
    - To make this analysable in Excel/Python, we write one CSV per `type`
      with `vals` expanded into numeric columns.

    Returns one representative output path (the type summary CSV).
    """
    if not parquet_path.exists():
        print(f"Parquet file does not exist, skipping: {parquet_path}")
        return None

    print("       Loading parquet and writing CSV outputs...")
    start = time.time()
    df = pl.read_parquet(parquet_path)

    required_cols = {"timestamp", "type", "vals"}
    if not required_cols.issubset(set(df.columns)):
        # Fallback: write a simple column-filtered CSV if this isn't tobii-munger's expected schema.
        pdf = pd.read_parquet(parquet_path)
        cols = [
            c
            for c in pdf.columns
            if any(
                key in c.lower()
                for key in ("time", "timestamp", "gaze", "eyeleft", "eyeright", "pupil")
            )
        ]
        if not cols:
            print(f"       Unrecognized schema and no matching columns in {parquet_path}, skipping.")
            return None
        cleaned_path = parquet_path.with_suffix("").with_name(parquet_path.stem + "_cleaned.csv")
        pdf[cols].to_csv(cleaned_path, index=False)
        elapsed = time.time() - start
        print(f"       Wrote {cleaned_path.name} in {elapsed:.1f}s")
        return cleaned_path

    out_dir = parquet_path.parent

    # 1) Write a small summary so you can quickly see what's inside.
    type_summary = (
        df.group_by("type")
        .len()
        .sort("len", descending=True)
    )
    summary_path = out_dir / f"{parquet_path.stem}_types.csv"
    type_summary.write_csv(summary_path)

    # 2) Write one CSV per type, expanding vals -> value_0, value_1, ...
    types = df.select("type").unique().to_series().to_list()
    for t in types:
        safe_type = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(t))
        sub = df.filter(pl.col("type") == t).select(["timestamp", "vals"])
        max_len = sub.select(pl.col("vals").list.len().max()).item()
        if max_len is None:
            continue
        expanded = sub.with_columns(
            [pl.col("vals").list.get(i).alias(f"value_{i}") for i in range(int(max_len))]
        ).drop("vals")
        out_path = out_dir / f"{parquet_path.stem}_{safe_type}.csv"
        expanded.write_csv(out_path)

    elapsed = time.time() - start
    print(f"       Wrote CSVs (summary + per-type) in {elapsed:.1f}s")
    return summary_path


def main() -> None:
    base_dir = get_base_dir()
    start_total = time.time()
    exports_root = base_dir / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Unpacking .g3 archives")
    print("=" * 60)
    unpacked_dirs = list(unpack_g3_archives(base_dir, exports_root))

    # 2. Look for directories that already contain gazedata.gz, imudata.gz, and scenevideo.mp4
    candidate_dirs = set(find_candidate_data_dirs(base_dir))
    for d in unpacked_dirs:
        candidate_dirs.update(find_candidate_data_dirs(d))

    if not candidate_dirs:
        print("No Tobii data directories found (with gazedata.gz / imudata.gz / scenevideo.mp4).")
        return

    sorted_candidates = sorted(candidate_dirs)
    total = len(sorted_candidates)
    print("\n" + "=" * 60)
    print(f"STEP 2: Converting {total} recording(s) to parquet + cleaned CSV")
    print("=" * 60)
    print("Found candidate data directories:")
    for d in sorted_candidates:
        print(" -", d)

    # 3. For each candidate dir, run tobii-munger and then process the output
    convert_times: list[float] = []
    for i, data_dir in enumerate(sorted_candidates, 1):
        elapsed_so_far = time.time() - start_total
        remaining = total - i
        if convert_times:
            avg_min = sum(convert_times) / len(convert_times) / 60
            est_remaining = avg_min * remaining
            print(f"\n--- Recording {i}/{total} | Elapsed: {elapsed_so_far/60:.1f} min | ~{est_remaining:.0f} min left ---")
        else:
            print(f"\n--- Recording {i}/{total} | Elapsed: {elapsed_so_far/60:.1f} min ---")
        rel = data_dir.relative_to(base_dir)
        out_parquet = exports_root / rel / "unified.parquet"

        if out_parquet.exists():
            print(f"       Parquet exists, skipping convert")
        else:
            step_start = time.time()
            ok = run_tobii_munger_convert(data_dir, out_parquet)
            if ok:
                convert_times.append(time.time() - step_start)
            if not ok:
                continue

        process_unified_parquet(out_parquet)

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 60)
    print(f"DONE in {total_elapsed/60:.1f} min total")
    print("=" * 60)


if __name__ == "__main__":
    main()

