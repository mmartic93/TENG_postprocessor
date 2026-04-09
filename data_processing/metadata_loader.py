import os
import pandas as pd
from typing import List, Dict
from server.config import ALLOWED_META_EXT, REQUIRED_META_COLUMNS


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
    else:
        raise ValueError('Unsupported metadata extension: ' + ext)
    return df


def format_metadata_rows(df: pd.DataFrame) -> List[Dict[str, str]]:
    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        rows.append({
            'row_number': index + 1,
            'ExpId': row.get('ExpId', ''),
            'TribuId': row.get('TribuId', ''),
            'RloadId': row.get('RloadId', ''),
            'DaqFile': row.get('DaqFile', ''),
            'MotorFile': row.get('MotorFile', ''),
        })
    return rows


def get_required_columns() -> List[str]:
    return REQUIRED_META_COLUMNS


def get_rows_for_tribuid(df: pd.DataFrame, tribuid: str) -> pd.DataFrame:
    if 'TribuId' not in df.columns:
        raise ValueError('Missing TribuId column in metadata')
    filtered = df[df['TribuId'].astype(str).str.strip() == tribuid.strip()]
    return filtered.reset_index(drop=True)


def find_loads_description_file(base_dir: str) -> str:
    for filename in ['LoadsDescription.ods', 'LoadsDescription.csv']:
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


def lookup_load_info(loads_df: pd.DataFrame, rload_id: str) -> Dict[str, str]:
    if not rload_id or str(rload_id).strip() == '':
        return {'Req': '', 'Gain': '', 'missing': False}
    rload_id_norm = str(rload_id).strip()
    matched = loads_df[loads_df['RloadId'] == rload_id_norm]
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


def get_paired_files(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Return DAQ and Motor file pairs from metadata, preserving row order."""
    pairs = []
    for index, row in df.iterrows():
        # Safely extract values, checking both column existence and null values
        daq = ''
        if 'DaqFile' in df.columns:
            val = row['DaqFile']
            if pd.notna(val) and str(val).strip():
                daq = str(val).strip()
        
        motor = ''
        if 'MotorFile' in df.columns:
            val = row['MotorFile']
            if pd.notna(val) and str(val).strip():
                motor = str(val).strip()
        
        rload_id = ''
        if 'RloadId' in df.columns:
            val = row['RloadId']
            if pd.notna(val) and str(val).strip():
                rload_id = str(val).strip()
        
        exp_id = ''
        if 'ExpId' in df.columns:
            val = row['ExpId']
            if pd.notna(val) and str(val).strip():
                exp_id = str(val).strip()
        
        pairs.append({
            'daq': daq.replace('\\', '/'),
            'motor': motor.replace('\\', '/'),
            'exp_id': exp_id,
            'rload_id': rload_id,
        })
    return pairs
