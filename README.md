# TDMS/CSV Metadata Viewer

This is a local Flask-based web app for loading metadata files, resolving relative DAQ/Motor paths, and plotting data from TDMS and CSV files.

## Required files

- A metadata file in the same folder as your data directory:
  - `*.csv` or `*.ods`
- The metadata file must include these columns:
  - `ExpId`
  - `TribuId`
  - `RloadId`
  - `DaqFile`
  - `MotorFile`
- A sidecar loads description file in the same folder as the metadata file:
  - `LoadsDescription.ods` or `LoadsDescription.csv`
- The `LoadsDescription` file must contain at least:
  - `RloadId`
  - `Req`
  - `Gain`

## How it works

1. Open the app in Chrome at `http://127.0.0.1:5000`
2. Enter the full path to your metadata file and click `Load`
3. The metadata preview page shows rows from the file
4. Enter the `TribuId` string you want to inspect
5. The app lists all experiments for that `TribuId`
6. For each experiment it shows:
   - `ExpId`
   - `RloadId`
   - `Req`
   - `Gain`
   - linked DAQ and Motor files
7. Click `View` on a DAQ or Motor file to open an interactive plot

## Special behavior

- DAQ files are expected to be voltage measurements
- The app multiplies DAQ data by the `Gain` value from `LoadsDescription`
- If `RloadId` is missing from `LoadsDescription`, the row is marked as missing
- The plot page shows the active downsampling percentage (default 80%)
- Power plots calculate `P = V^2 / R` where V is gain-scaled voltage and R is `Req`
- Mean power is calculated as the average power across the entire DAQ file
- The file list page shows a scatter plot of Mean Power vs Resistance (Req) for the selected `TribuId`

## Dependencies

This project requires Python and the following packages in the virtual environment:

- `Flask`
- `pandas`
- `plotly`
- `nptdms`
- `odfpy` (for `.ods` support)

## Installation

1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install required packages:
   ```powershell
   .\.venv\Scripts\pip.exe install flask pandas plotly nptdms odfpy
   ```
3. Run the app:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```
4. Visit `http://127.0.0.1:5000` in Chrome.

## Notes

- Use the full metadata path so relative file links resolve against the correct folder.
- The app currently resolves file paths relative to the metadata file location.
- If a file is missing, the UI will display an error for that row.
