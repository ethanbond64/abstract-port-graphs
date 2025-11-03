import abc
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Type, List, Tuple

from arc.arc_perception import adjacency
from synthesis.design_time_models import PerceivedValue
from synthesis.program_state import PerceptionModel
from synthesis.arc_specific.perception_generation import apply_perception
from synthesis.program_state import ProgramState, EvaluationContext
from synthesis.synthesis_engine import SynthesisDirective


class ActionType(Enum):
    PERCEPTION = 0
    SYNTHESIS = 1


class Action(abc.ABC):

    @abstractmethod
    def get_action_type(self) -> ActionType:
        ...

    @abstractmethod
    def run(self, program: ProgramState) -> Any:
        ...


class PerceptionAction(Action):

    def get_action_type(self) -> ActionType:
        return ActionType.PERCEPTION

    def run(self, program_state: ProgramState) -> Tuple[PerceptionModel, List[PerceivedValue]]:
        perception_model = PerceptionModel([adjacency])
        challenge = program_state.get_challenge()
        perception_values_by_type: Dict[Type, Dict[int, PerceivedValue]] = apply_perception(challenge.train,
                                                                                            perception_model,
                                                                                            program_state.get_uvn())
        perception_value_list = [value for id_dict in perception_values_by_type.values() for value in id_dict.values()]

        test_perception_values_by_type: Dict[Type, Dict[int, PerceivedValue]] = apply_perception(challenge.test,
                                                                                            perception_model,
                                                                                            None, test=True)
        test_perception_value_list = [value for id_dict in test_perception_values_by_type.values()
                                      for value in id_dict.values()]

        program_state.get_perception_xref().record_perception(perception_model, perception_value_list, test_perception_value_list)
        return perception_model, perception_value_list


class SynthesisAction(Action):

    def __init__(self, directives: List[SynthesisDirective], evaluation_context: EvaluationContext):
        self.directives: List[SynthesisDirective] = directives
        self.evaluation_context: EvaluationContext = evaluation_context

    def get_action_type(self) -> ActionType:
        return ActionType.SYNTHESIS

    def run(self, program_state: ProgramState) -> Any:
        collected_results = []
        for directive in self.directives:
            result = directive.run(program_state, self.evaluation_context)
            collected_results.append((directive, result))
        return collected_results


# Future actions:
# Individual directive
# Re-evaluation of existing program with different eval context



