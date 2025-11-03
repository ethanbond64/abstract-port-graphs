from collections import defaultdict
from typing import Dict, Type, List, DefaultDict

from arc.arc_utils import Case
from arc.arc_objects import EnvironmentShape, ArcObject, apply_object_def_functions_iterative
from synthesis.unified_value_network import UnifiedValueNetwork
from synthesis.design_time_models import PerceivedValue
from synthesis.program_state import PerceptionModel


def apply_perception(cases: List[Case], perception_model: PerceptionModel,
                     uvn: UnifiedValueNetwork, test=False) -> Dict[Type, Dict[int, PerceivedValue]]:

    # Populate uvn
    perceived_value_id_lookup: DefaultDict[Type, Dict[int, PerceivedValue]] = defaultdict(dict)

    def track_perceived_value(value: PerceivedValue, lookup=perceived_value_id_lookup, uvn_arg=uvn):
        lookup[value.value_type][value.id] = value
        if uvn_arg is not None:
            uvn_arg.add_perceived_value(value)

    for case_index, case in enumerate(cases):

        # env shapes
        input_env_shape = EnvironmentShape.of(case.input_matrix)
        input_perceived_env = PerceivedValue(EnvironmentShape, input_env_shape, perception_model.id, case_index,
                                             test=test)
        track_perceived_value(input_perceived_env)

        # Output is only available in training cases
        if not test:
            output_env_shape = EnvironmentShape.of(case.output_matrix)
            output_perceived_env = PerceivedValue(EnvironmentShape, output_env_shape, perception_model.id, case_index,
                                                  output=True)
            track_perceived_value(output_perceived_env)

        # Objects
        for input_object in apply_object_def_functions_iterative(perception_model.perception_functions,
                                                                 case.input_matrix + 1, perception_model.bg_color):
            if input_object.get_perception_function() in perception_model.perception_functions:
                input_perceived_object = PerceivedValue(ArcObject, input_object, perception_model.id, case_index,
                                                        test=test)
                track_perceived_value(input_perceived_object)

        # Output is only available in training cases
        if not test:
            for output_object in apply_object_def_functions_iterative(perception_model.perception_functions,
                                                                      case.output_matrix + 1, perception_model.bg_color):
                if output_object.get_perception_function() in perception_model.perception_functions:
                    output_perceived_object = PerceivedValue(ArcObject, output_object, perception_model.id, case_index,
                                                             output=True)
                    track_perceived_value(output_perceived_object)

    return perceived_value_id_lookup
