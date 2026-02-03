import os
import pickle
import time
from pathlib import Path
from typing import Any, Union


def robust_pickle_load(
    path: Union[str, Path], retries: int = 5, backoff: float = 0.1
) -> Any:
    """Load a pickle file with simple retry logic for transient write races.

    Args:
        path:
            File path to load.
        retries:
            Number of attempts before failing.
        backoff:
            Base sleep seconds between attempts (linear backoff).

    Returns:
        The unpickled Python object.

    Raises:
        pickle.UnpicklingError | EOFError | FileNotFoundError | KeyError:
            If load fails after all retries.
    """

    path_obj = Path(path)
    for attempt in range(retries):
        try:
            with path_obj.open("rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, KeyError, FileNotFoundError):
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise


def atomic_pickle_dump(obj: Any, path: Union[str, Path]) -> None:
    """Atomically write a pickle file.

    Writes to a temporary file in the same directory, fsyncs, and replaces
    the target path to avoid readers observing partial writes.

    Args:
        obj:
            Object to pickle.
        path:
            Destination file path.
    """

    path_obj = Path(path)
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path_obj.with_name(path_obj.name + f".tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replace
        tmp_path.replace(path_obj)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
