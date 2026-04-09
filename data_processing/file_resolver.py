import os


def resolve_relative_path(base_dir: str, relative_path: str) -> str:
    # Normalize forward slashes to backslashes on Windows
    normalized_rel = relative_path.replace('/', os.sep)
    target = os.path.normpath(os.path.join(base_dir, normalized_rel))
    base_norm = os.path.normpath(base_dir)
    try:
        if os.path.commonpath([base_norm, target]) != base_norm:
            raise ValueError('Path escapes metadata directory')
    except ValueError as e:
        if 'different drives' in str(e):
            raise ValueError('Path escapes metadata directory')
        raise
    return target


def file_exists(path: str) -> bool:
    return os.path.exists(path)


def normalize_display_path(path: str) -> str:
    """Convert backslashes to forward slashes for consistent display."""
    return path.replace('\\', '/')
