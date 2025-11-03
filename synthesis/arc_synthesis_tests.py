import time
import unittest

import numpy as np

from arc.arc_objects import ArcInterpreter
from arc.arc_utils import read_challenges, read_solutions
from synthesis.meta_learning import MetaModel, ActionLogger
from synthesis.synthesis_loop import synthesis_loop

# Example subset that can be solver in ~1s or less
CHALLENGE_ID_SUBSET = [
    "0962bcdd",
    "137eaa0f",
    "1bfc4729",
    "1caeab9d",
    "1cf80156",
    "48d8fb45",
    "5521c0d9",
    "95990924",
    "a61ba2ce",
    "b1948b0a",
    "c8f0f002",
    "f76d97a5"
]


def solve_wrapper(graph, input_matrix, output_matrix):
    in_mat = input_matrix + 1
    sol = ArcInterpreter().solve(graph, in_mat) - 1
    equal = np.array_equal(sol, np.array(output_matrix))
    if not equal:
        print(sol)
        print(np.array(output_matrix))
    return equal


class CompletionTests(unittest.TestCase):

    def test_training_challenges(self):
        challenges = read_challenges("data/arc-agi_training_challenges.json")
        solutions = read_solutions("data/arc-agi_training_solutions.json")

        for challenge_id in CHALLENGE_ID_SUBSET:
            with self.subTest(challenge=challenge_id):
                print("BEGIN CHALLENGE", challenge_id)
                challenge = challenges[challenge_id]

                start = time.time()
                graph = synthesis_loop(MetaModel(), ActionLogger("log.json"), challenge)
                end = time.time()
                print("Total time", end - start)

                passing = True
                challenge_solutions = solutions[challenge.challenge_id]
                for case_idx, case in enumerate(challenge.test):
                    if not solve_wrapper(graph, case.input_matrix, challenge_solutions[case_idx]):
                        passing = False

                self.assertTrue(passing)


if __name__ == "__main__":
    unittest.main()
