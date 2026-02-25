from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# Root folder that contains your Tobii recordings
BASE_DIR = Path("/Users/ravi/Desktop/tobi copy")


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
    for g3_file in root.rglob("*.g3"):
        # Only try to treat it as a zip if it actually is one
        if not zipfile.is_zipfile(g3_file):
            print(f"Skipping non-zip .g3 file: {g3_file}")
            continue

        target_dir = out_root / (g3_file.stem + "_raw")
        if target_dir.exists():
            print(f"Already unpacked, skipping: {target_dir}")
            yield target_dir
            continue

        print(f"Unpacking {g3_file} -> {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(g3_file, "r") as zf:
            zf.extractall(target_dir)

        yield target_dir


def run_tobii_munger_convert(data_dir: Path, out_parquet: Path) -> bool:
    """
    Call `python -m tobii_munger.convert` on data_dir to produce out_parquet.

    Returns True on success, False on failure.
    """
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "tobii_munger.convert",
        str(data_dir),
        str(out_parquet),
    ]
    print("Running:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"tobii-munger failed for {data_dir}: {exc}")
        return False


def process_unified_parquet(parquet_path: Path) -> Optional[Path]:
    """
    Load a unified Parquet file and write out a 'cleaned' CSV that focuses
    on time and gaze-related columns. Returns the cleaned CSV path.
    """
    if not parquet_path.exists():
        print(f"Parquet file does not exist, skipping: {parquet_path}")
        return None

    print(f"Loading unified data from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Keep time and gaze-related columns
    cols = [
        c
        for c in df.columns
        if any(
            key in c.lower()
            for key in ("time", "timestamp", "gaze", "eyeleft", "eyeright", "pupil")
        )
    ]

    if not cols:
        print(f"No matching gaze/time columns found in {parquet_path}, skipping.")
        return None

    cleaned = df[cols]
    cleaned_path = parquet_path.with_suffix("").with_name(
        parquet_path.stem + "_cleaned.csv"
    )
    cleaned.to_csv(cleaned_path, index=False)
    print(f"Wrote cleaned CSV to {cleaned_path}")
    return cleaned_path


def main() -> None:
    exports_root = BASE_DIR / "exports"
    exports_root.mkdir(parents=True, exist_ok=True)

    # 1. Unpack any .g3 archives we find
    unpacked_dirs = list(unpack_g3_archives(BASE_DIR, exports_root))

    # 2. Look for directories that already contain gazedata.gz, imudata.gz, and scenevideo.mp4
    candidate_dirs = set(find_candidate_data_dirs(BASE_DIR))
    for d in unpacked_dirs:
        candidate_dirs.update(find_candidate_data_dirs(d))

    if not candidate_dirs:
        print("No Tobii data directories found (with gazedata.gz / imudata.gz / scenevideo.mp4).")
        return

    print("Found candidate data directories:")
    for d in sorted(candidate_dirs):
        print(" -", d)

    # 3. For each candidate dir, run tobii-munger and then process the output
    for data_dir in sorted(candidate_dirs):
        rel = data_dir.relative_to(BASE_DIR)
        out_parquet = exports_root / rel / "unified.parquet"

        if out_parquet.exists():
            print(f"Unified parquet already exists, skipping convert: {out_parquet}")
        else:
            ok = run_tobii_munger_convert(data_dir, out_parquet)
            if not ok:
                continue

        process_unified_parquet(out_parquet)


if __name__ == "__main__":
    main()

