from pathlib import Path
from typing import Dict, List
from multiprocessing import Queue, Process

import numpy as np

from arc.arc_objects import ArcInterpreter
from arc.arc_utils import ArcTask, read_challenge_text
from synthesis.meta_learning import MetaModel, ActionLogger
from synthesis.synthesis_loop import synthesis_loop


def synthesis_loop_worker(queue: Queue, meta_model: MetaModel, logger: ActionLogger, task: ArcTask):
    """Worker function to run synthesis_loop in a separate process."""
    try:
        result = synthesis_loop(meta_model, logger, task)
        queue.put(result)
    except Exception:
        queue.put(None)


def synthesis_loop_with_timeout(meta_model: MetaModel, logger: ActionLogger, task: ArcTask, timeout):
    queue = Queue()
    process = Process(target=synthesis_loop_worker, args=(queue, meta_model, logger, task))

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return None

    if not queue.empty():
        return queue.get()

    return None


def solve_private_set(file_path: str, task_timeout_seconds: int) -> Dict[str, List[Dict[str, List[List[int]]]]]:

    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File {path_obj} not found")

    challenges: Dict[str, ArcTask] = read_challenge_text(path_obj.read_text(encoding="utf-8"))

    # Iterate challenges
    output = {}

    meta_model = MetaModel()
    logger = ActionLogger("")
    for task in challenges.values():

        arc_program = synthesis_loop_with_timeout(meta_model, logger, task, task_timeout_seconds)
        test_results = []
        if arc_program is not None:

            for test in task.test:
                interpreter = ArcInterpreter()
                output_grid: np.ndarray = interpreter.evaluate_program(arc_program, test.input_matrix + 1)
                output_grid = output_grid - 1
                test_results.append({"attempt_1": output_grid.astype(int).tolist(), "attempt_2": [[0]]})
        else:
            for _ in task.test:
                test_results.append({"attempt_1": [[0]], "attempt_2": [[0]]})

        output[task.challenge_id] = test_results

    return output
