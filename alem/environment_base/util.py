import bz2
import os
import pickle
import tempfile
from typing import Any


def save_compressed_pickle(title: str, data: Any):
    """Atomically serialize an object to a bzip2-compressed pickle.

    Args:
        title: Destination file path.
        data: Python object to serialize.
    """
    # Write to a temp file next to the target, then atomically rename.
    # This prevents concurrent jobs from reading a half-written cache file.
    dir_name = os.path.dirname(os.path.abspath(title))
    with tempfile.NamedTemporaryFile(dir=dir_name, suffix=".tmp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with bz2.BZ2File(tmp_path, "w") as f:
            pickle.dump(data, f)
        os.replace(tmp_path, title)  # atomic on Linux
    except Exception:
        os.unlink(tmp_path)
        raise


def load_compressed_pickle(file: str):
    """Load an object from a bzip2-compressed pickle.

    Args:
        file: Compressed pickle path.

    Returns:
        Deserialized Python object.
    """
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data
