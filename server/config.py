import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_META_EXT = {'.csv', '.ods', '.xlsx', '.xls'}
REQUIRED_META_COLUMNS = ['TribuId', 'RloadId', 'SampleIdTriboNeg', 'SampleIdTriboPos', 'Date']
MAX_PREVIEW_ROWS = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
