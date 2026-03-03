## Tobii data processing helper

This small Python project automates converting Tobii Glasses recordings into tabular data you can analyse in Python.

It is designed around the open-source `tobii-munger` project, which converts raw Tobii Glasses 3 recordings into a single unified Parquet file. This repo then shows how to batch-run that conversion and optionally turn the result into CSV.

### 1. Setup

From a terminal:

```bash
cd /Users/ravi/Desktop/labwork
python -m venv .venv
source .venv/bin/activate  # on macOS / Linux

pip install -r requirements.txt
```

You also need two non-Python tools that `tobii-munger` depends on:

- `ffmpeg`
- `jq`

On macOS with Homebrew you can install them with:

```bash
brew install ffmpeg jq
```

### 2. Folder layout

By default the script expects your Tobii project folder to be:

- `/Users/ravi/Desktop/tobi copy`

You can change this path inside `process_tobii.py` if needed.

The script will look for:

- Tobii Glasses archive files such as `.g3`, and
- Any subdirectories that already contain `gazedata.gz`, `imudata.gz`, and `scenevideo.mp4`.

### 3. Running the processor

Activate the virtual environment if it is not already active:

```bash
cd /Users/ravi/Desktop/labwork
source .venv/bin/activate
```

Then run:

```bash
python process_tobii.py
```

What this does:

- Recursively scans `/Users/ravi/Desktop/tobi copy`
- For each `.g3` file:
  - Tries to unpack it
  - Locates the directory that contains `gazedata.gz`, `imudata.gz`, and `scenevideo.mp4`
  - Calls `python -m tobii_munger.convert` on that directory to create a unified `.parquet` file
- Optionally converts that Parquet file into a "clean" CSV focusing on timing and gaze-related columns.

All output is written under an `exports/` folder inside `tobi copy`.

### 4. Customising the analysis

The example analysis is intentionally minimal and just demonstrates:

- Loading the unified Parquet file
- Selecting time and gaze-related columns
- Saving them as a cleaned CSV

You can modify `process_tobii.py` (the `process_unified_parquet` function) to perform any additional calculations you want (aggregations, visualisations, etc.).

