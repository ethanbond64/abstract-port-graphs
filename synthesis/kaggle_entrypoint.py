from pathlib import Path
from typing import Dict, List

import numpy as np

from arc.arc_objects import ArcInterpreter
from arc.arc_utils import ArcTask, read_challenge_file
from synthesis.meta_learning import MetaModel, ActionLogger
from synthesis.synthesis_loop import synthesis_loop


def solve_private_set(file_path: str) -> Dict[str, List[Dict[str, List[List[int]]]]]:

    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File {path_obj} not found")

    challenges: Dict[str, ArcTask] = read_challenge_file(path_obj.read_text(encoding="utf-8"))

    # Iterate challenges
    output = {}

    meta_model = MetaModel()
    logger = ActionLogger("")
    for task in challenges.values():

        arc_program = synthesis_loop(meta_model, logger, task)
        test_results = []
        if arc_program is not None:

            for test in task.test:
                interpreter = ArcInterpreter()
                output_grid: np.ndarray = interpreter.evaluate_program(arc_program, test.input_matrix + 1)
                output_grid = output_grid - 1
                test_results.append({"attempt_1": output_grid.tolist(), "attempt_2": test.input_matrix})
        else:
            for _ in task.test:
                test_results.append({"attempt_1": [[0]], "attempt_2": [[0]]})

        output[task.challenge_id] = test_results

    return output
