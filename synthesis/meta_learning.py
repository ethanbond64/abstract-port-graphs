import json
import random
from typing import List, Tuple, Any

from synthesis.action_space import choose_from_all_actions
from synthesis.execution import Action
from synthesis.goals import GoalGraph, Goal, OperatorSynthesisGoal, PredicateSynthesisGoal
from synthesis.program_state import ProgramState


class MetaModel:

    def __init__(self):
        self.weights = {} # TODO read from persisted file w/ program args etc.

    def decide_action(self, state: ProgramState, current_goals: List[GoalGraph]) -> Tuple[Goal, Action]:
        if len(current_goals) == 0:
            return None, None
        chosen_goal: Goal = current_goals[0]

        if isinstance(chosen_goal, (OperatorSynthesisGoal)):
            program_subset = list(filter(lambda t: t.root_value_type == chosen_goal.root_value_type,
                                         state.get_trace_graph().get_all_traces()))
            action = choose_from_all_actions(state, chosen_goal.evaluation_context, program_subset)
        else:
            actions: List[Action] = chosen_goal.get_actions()
            action = random.choice(actions) if len(actions) > 0 else None
        return chosen_goal, action

    def update_beliefs(self, output, logger: 'ActionLogger'):
        pass


class ActionLogger:

    def __init__(self, file_path: str):
        self.__file_path: str = file_path
        self.__log_entries: List[dict] = []
        self.__invocation_count: int = 0
        self.__flush_interval: int = 10
        self.__logging_active = False

    def log_action(self, goal: Goal, action: Action, result: Any):

        if not self.__logging_active:
            return

        visited = set()
        log_entry = {
            'goal': self.__to_dict(goal, visited),
            'action': self.__to_dict(action, visited),
            'result': self.__to_dict(result, visited)
        }
        self.__log_entries.append(log_entry)
        self.__invocation_count += 1

        if self.__invocation_count % self.__flush_interval == 0:
            self.__flush_to_file()

    def __to_dict(self, obj: Any, visited: set) -> dict:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, type):
            return str(obj)

        obj_id = id(obj)
        if obj_id in visited:
            return {'__circular_ref__': obj_id}

        if isinstance(obj, (list, tuple)):
            visited.add(obj_id)
            result = [self.__to_dict(item, visited) for item in obj]
            visited.remove(obj_id)
            return result

        if isinstance(obj, dict):
            visited.add(obj_id)
            result = {}
            for k, v in obj.items():
                if isinstance(k, (str, int, float, bool)) or k is None:
                    key = k
                else:
                    key = str(k)
                result[key] = self.__to_dict(v, visited)
            visited.remove(obj_id)
            return result

        if hasattr(obj, '__dict__'):
            visited.add(obj_id)
            result = {
                '__class__': obj.__class__.__name__,
                '__id__': obj_id,
                **{k: self.__to_dict(v, visited) for k, v in obj.__dict__.items()
                   if not (isinstance(k, str) and k.startswith('__') and k.endswith('__'))}
            }
            visited.remove(obj_id)
            return result

        return str(obj)

    def __flush_to_file(self):
        if not self.__log_entries:
            return

        try:
            with open(self.__file_path, 'a') as f:
                for entry in self.__log_entries:
                    f.write(json.dumps(entry) + '\n')
            self.__log_entries.clear()
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def flush(self):
        self.__flush_to_file()
