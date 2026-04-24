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


def validate_tribuid(tribuid_input: str, df) -> str:
    if not tribuid_input or not tribuid_input.strip():
        raise ValueError('TribuId is required')

    if 'TribuId' not in df.columns:
        raise ValueError('Metadata does not contain a TribuId column')

    # Split the input into a list of targets
    target_ids = [id.strip() for id in tribuid_input.split(',') if id.strip()]
    valid_ids = df['TribuId'].astype(str).str.strip().tolist()

    # Validate each ID
    for tid in target_ids:
        if tid not in valid_ids:
            raise ValueError(f'TribuId "{tid}" not found in metadata')

    # Return the clean, comma-separated string back
    return ", ".join(target_ids)
