from pathlib import Path


def get_base_path():

    base_path = Path(__file__).resolve().parent
    return base_path

def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path from the project root."""
    return get_base_path()/relative_path