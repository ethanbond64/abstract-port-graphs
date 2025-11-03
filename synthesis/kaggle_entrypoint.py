from pathlib import Path
from typing import Dict

from arc.arc_utils import ArcTask, read_challenge_file


def solve_private_set(file_path: str):

    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File {path_obj} not found")

    challenges: Dict[str, ArcTask] = read_challenge_file(path_obj.read_text(encoding="utf-8"))

    # Iterate challenges

    # Give each a variable timeout, run on another thread, Get the graph

    # Apply to each test input if possible
    # Add to the collection

    # default to the input

    # Continue