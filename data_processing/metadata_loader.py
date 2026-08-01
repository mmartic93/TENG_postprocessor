import os
import pandas as pd
from typing import List, Dict
from server.config import ALLOWED_META_EXT, REQUIRED_META_COLUMNS

from datetime import datetime

def make_date_token(s: str) -> str:
    """Convert various date strings to 'DDMMYYYY_HHMMSS'"""
    if s is None:
        raise ValueError('Empty date string')
    s_norm = ' '.join(str(s).split())

    # Try a set of common explicit formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s_norm, fmt)
            return dt.strftime("%d%m%Y_%H%M%S")
        except ValueError:
            continue

    raise Exception("Unable to parse date string: %s", s)


def allowed_meta(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_META_EXT


def parse_metadata_csv(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == '.csv':
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif ext == '.ods':
        try:
            df = pd.read_excel(path, engine='odf', dtype=str)
        except ImportError as error:
            raise RuntimeError('ODS support requires odfpy: install via pip install odfpy') from error
        except Exception as error:
            raise RuntimeError(f'Unable to parse ODS metadata: {error}') from error
        df = df.fillna('')
    elif ext in ('.xlsx', '.xls'):
        try:
            # Let pandas choose the appropriate engine (openpyxl for .xlsx, xlrd for .xls)
            df = pd.read_excel(path, dtype=str)
        except ImportError as error:
            raise RuntimeError('Excel support requires openpyxl (for .xlsx) and/or xlrd (for .xls): pip install openpyxl xlrd') from error
        except Exception as error:
            raise RuntimeError(f'Unable to parse Excel metadata: {error}') from error
        df = df.fillna('')
    else:
        raise ValueError('Unsupported metadata extension: ' + ext)
    return df


def format_metadata_rows(df: pd.DataFrame) -> List[Dict[str, str]]:
    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        rows.append({
            'row_number': index + 1,
            'TribuId': row.get('TribuId', ''),
            'RloadId': row.get('RloadId', ''),
            'SampleIdTriboNeg': row.get('SampleIdTriboNeg', ''),
            'SampleIdTriboPos': row.get('SampleIdTriboPos', ''),
            'Date': row.get('Date', '')
        })
    return rows


def get_required_columns() -> List[str]:
    return REQUIRED_META_COLUMNS


def get_rows_for_tribuid(df: pd.DataFrame, tribuid_input: str) -> pd.DataFrame:
    """
    Handles both a single ID and a comma-separated string of IDs.
    """
    if 'TribuId' not in df.columns:
        raise ValueError('Missing TribuId column in metadata')

    # 1. Split the string by comma and remove whitespace
    # If tribuid_input is "ID1, ID2", this creates ['ID1', 'ID2']
    target_ids = [id.strip() for id in tribuid_input.split(',') if id.strip()]

    # 2. Use .isin() to filter for all IDs in the list
    filtered = df[df['TribuId'].astype(str).str.strip().isin(target_ids)]

    if filtered.empty:
        raise ValueError(f'None of the TribuIds {target_ids} were found in metadata')

    return filtered.reset_index(drop=True)


def find_loads_description_file(base_dir: str) -> str:
    for filename in ['LoadsDescription.ods', 'LoadsDescription.xlsx', 'LoadsDescription.xls', 'LoadsDescription.csv']:
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError('LoadsDescription file not found in metadata folder')


def load_loads_description(path: str) -> pd.DataFrame:
    df = parse_metadata_csv(path)
    if 'RloadId' not in df.columns:
        raise ValueError('LoadsDescription file missing RloadId column')
    # Normalize keys for lookup
    df = df.copy()
    df['RloadId'] = df['RloadId'].astype(str).str.strip()
    return df


def lookup_load_info(loads_df: pd.DataFrame, RloadId: str) -> Dict[str, str]:
    if not RloadId or str(RloadId).strip() == '':
        return {'Req': '', 'Gain': '', 'missing': False}
    RloadId_norm = str(RloadId).strip()
    matched = loads_df[loads_df['RloadId'] == RloadId_norm]
    if matched.empty:
        return {'Req': '', 'Gain': '', 'missing': True}

    req = ''
    gain = ''
    if 'Req' in matched.columns:
        req = str(matched.iloc[0]['Req']).strip()
    if 'Gain' in matched.columns:
        gain = str(matched.iloc[0]['Gain']).strip()
    return {'Req': req, 'Gain': gain, 'missing': False}


def validate_metadata_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in REQUIRED_META_COLUMNS)


def get_sample_range(df: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    return df.iloc[start_row - 1:end_row].reset_index(drop=True)


def collect_sample_files(df: pd.DataFrame) -> List[str]:
    daq_files = df['DaqFile'].astype(str).str.strip().replace('', pd.NA).dropna().tolist() if 'DaqFile' in df.columns else []
    motor_files = df['MotorFile'].astype(str).str.strip().replace('', pd.NA).dropna().tolist() if 'MotorFile' in df.columns else []
    return list(dict.fromkeys(daq_files + motor_files))


def get_experiment_folders(df: pd.DataFrame, raw_dir: str) -> List[dict]:
    """Reconstruct experiment folder paths based on metadata.

    Assumes there is a `RawData` directory in the same folder as the metadata file.
    For each row the function attempts to locate folders under:
      RawData/<TribuId>/<SamplePair>/... (any depth)
    and collects all subfolders that look like experiment folders (commonly containing
    the RloadId as a suffix like '-100' or date-time prefixes).

    Returns a list of dictionary representing the experiment folders.
    """
    results: List[dict] = []

    for index, row in df.iterrows():
        TribuId = str(row.get('TribuId', '') or '').strip()
        RloadId = str(row.get('RloadId', '') or '').strip()
        sample_neg = str(row.get('SampleIdTriboNeg', '') or '').strip()
        sample_pos = str(row.get('SampleIdTriboPos', '') or '').strip()
        date = str(row.get('Date', '') or '').strip()
        date = make_date_token(date)

        exp_path = os.path.join(raw_dir, TribuId, f"{sample_neg}-{sample_pos}", f"{date}-{RloadId}")
        if not os.path.exists(exp_path):
            raise Exception("Error, the expected exp_path is not valid: %s", exp_path)
        else:
            results.append({
                'TribuId': TribuId,
                'RloadId': RloadId,
                'SampleIdTriboNeg': sample_neg,
                'SampleIdTriboPos': sample_pos,
                'Date': date,
                'exp_path': exp_path,
            })
    return results