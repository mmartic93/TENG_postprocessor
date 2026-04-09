from typing import Tuple


def validate_row_range(start: str, end: str, max_rows: int) -> Tuple[int, int]:
    if not start or not end:
        raise ValueError('Start row and end row are required')
    try:
        start_row = int(start)
        end_row = int(end)
    except ValueError:
        raise ValueError('Start and end row must be integers')
    if start_row < 1 or end_row < 1:
        raise ValueError('Row numbers must be positive')
    if start_row > end_row:
        raise ValueError('Start row must be less than or equal to end row')
    if end_row > max_rows:
        raise ValueError(f'End row must be no greater than {max_rows}')
    return start_row, end_row

def validate_tribuid(tribuid: str, df) -> str:
    if not tribuid or not tribuid.strip():
        raise ValueError('TribuId is required')

    if 'TribuId' not in df.columns:
        raise ValueError('Metadata does not contain a TribuId column')

    trimmed = tribuid.strip()
    values = df['TribuId'].astype(str).str.strip()
    if trimmed not in values.tolist():
        raise ValueError(f'TribuId "{trimmed}" not found in metadata')
    return trimmed
