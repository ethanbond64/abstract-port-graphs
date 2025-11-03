from collections import defaultdict
from collections.abc import Callable
from itertools import product
from typing import DefaultDict, Set, Type, List, Tuple, Any, Dict

from arc.arc_objects import difference_orientation, ObjectIdentity, Orientation, ArcInterpreter
from nodes import OperatorNode
from functional_operator import FunctionalOperator
from arc.arc_operators import ApplyTwoDimensionalRelationshipOperator, \
    ApplyIdentityRelationshipOperator, TransformLocationRelationshipOperator, ApplyLinearPatternOperator, \
    CreateRectangleOperator, DrawLineOperator, DivideIdentityOperator, SpecificLocationValueOperator, \
    MeasureCentersOperator, MeasureClosestSidesOperator, MeasureIdentityDeltaOperator, \
    SumLocationRelationshipOperator, PossibleCutLocationsOperator, CutShapeOperator, DivideToZonesOperator, \
    GetCavityZonesOperator, GetOuterCavitiesOperator, GetZoneByIndexOperator, ZoneLocationToFullLocationOperator, \
    GetClusterColorPatternOperator, GetClusterRadialColorPatternOperator, GetLinearShapePatternOperator, \
    TranslateIdentityOperator, BinaryAndOperator, BinaryNorOperator, ShapeSizeEnvOperator, ProportionalEnvOperator, \
    MoveOperator, ApproachOperator, CollapseDirectionOperator, Rotate90Operator
from operator_primitives import SetRankOperator, ApplyScalarOpToSetOperator, \
    CreateLocalSetOperator, AddToLocalSetOperator, MultiplyOperator, AdditionOperator, SubtractionOperator, SumSetOperator
from synthesis.unified_value_network import UnifiedValueNetwork

# Use cases

# Constructor not included
OPERATOR_LIST: List[OperatorNode] = [
    SetRankOperator(),
    SetRankOperator(reverse=True),
    ApplyTwoDimensionalRelationshipOperator(),
    ApplyIdentityRelationshipOperator(),
    TransformLocationRelationshipOperator(),
    ApplyLinearPatternOperator(),
    CreateRectangleOperator(),
    DrawLineOperator(),
    DivideIdentityOperator(),
    SpecificLocationValueOperator(),
    # FindClosestOperator() SEARCH/CONDITIONAL ONLY
    MeasureCentersOperator(),
    MeasureClosestSidesOperator(),
    MeasureIdentityDeltaOperator(),
    SumLocationRelationshipOperator(),
    PossibleCutLocationsOperator(),
    CutShapeOperator(),
    ApplyScalarOpToSetOperator(),
    CreateLocalSetOperator(),
    AddToLocalSetOperator(),
    DivideToZonesOperator(),
    GetCavityZonesOperator(),
    GetOuterCavitiesOperator(),
    GetZoneByIndexOperator(),
    ZoneLocationToFullLocationOperator(),
    GetClusterColorPatternOperator(),
    GetClusterRadialColorPatternOperator(),
    GetLinearShapePatternOperator(),

    # Arithmetic
    MultiplyOperator(),
    AdditionOperator(),
    SubtractionOperator(),
    SumSetOperator(),

    # Binary
    BinaryAndOperator(),
    BinaryNorOperator(),

    # Env (non-primitive)
    ShapeSizeEnvOperator(),
    ProportionalEnvOperator(),

    # Physics
    MoveOperator(),
    ApproachOperator(),
    CollapseDirectionOperator(),

    # Identity transform (incomplete)
    Rotate90Operator(),
    TranslateIdentityOperator()
]

# 1. Get operator by output type
TYPE_TO_OPERATOR: DefaultDict[Type, Set[OperatorNode]] = defaultdict(set)

for op in OPERATOR_LIST:
    TYPE_TO_OPERATOR[op.output_type].add(op)


def default_output_validation(output_type: Type, output_value: Any) -> bool:
    return True

def apply_operator_safe(operator: OperatorNode, args: List[Any]) -> Any:

    if isinstance(operator, FunctionalOperator):
        temp_args = [ArcInterpreter(initial_depth=1)] + args
    else:
        temp_args = args
    return operator.apply(*temp_args)


class ArgumentSearchStrategy:

    # Default base implementation can have a validation that first checks if it is even possible
    # for the value to come from this operator
    def __init__(self, operator: OperatorNode, validate_possible: Callable[[Type, Any], bool] = default_output_validation):
        self.operator = operator
        self.validate_possible = validate_possible

    # Returns a list of valid combinations of args (lists of (type, value) all of the same len... signature len)
    def generate_argument_value_tuples(self, uvn: UnifiedValueNetwork, output_type: Type,
                                       output_value: Any, case_index: int) -> List[List[Tuple[Type, Any]]]:
        if not self.validate_possible(output_type, output_value):
            return []

        return self._generate_argument_value_tuples_inner(uvn, output_type, output_value, case_index)

    def _generate_argument_value_tuples_inner(self, uvn: UnifiedValueNetwork, output_type: Type,
                                              output_value: Any, case_index: int) -> List[List[Tuple[Type, Any]]]:
        ...


class BijectiveSearchStrategy(ArgumentSearchStrategy):

    def __init__(self, operator: OperatorNode, reverse_fn: Callable[[Any], Tuple], validate_possible = default_output_validation):
        super().__init__(operator, validate_possible)
        self.reverse_fn = reverse_fn

    def _generate_argument_value_tuples_inner(self, uvn: UnifiedValueNetwork,
                                              output_type: Type, output_value: Any, case_index: int) -> List[List[Tuple[Type, Any]]]:
        return self.reverse_fn(output_value)


# Functions with two args where if you have one arg value and the output value,
# and can calculate the other arg deterministically
class TwoArgSearchAndCalculate(ArgumentSearchStrategy):

    def __init__(self, operator: OperatorNode, search_index: int, secondary_reverse_fns: List[Callable[[Any, Any], Any]],
                 validate_possible = default_output_validation):
        super().__init__(operator, validate_possible)

        if search_index not in {0, 1}:
            raise ValueError("Search index must be 0 or 1.")

        self.search_index = search_index
        self.secondary_reverse_fns = secondary_reverse_fns

    def _generate_argument_value_tuples_inner(self, uvn: UnifiedValueNetwork, output_type: Type,
                                              output_value: Any, case_index: int) -> List[List[Tuple[Type, Any]]]:
        # Iterate search index arg known values from the uvn
        arg_lists = []
        search_type = self.operator.input_types[self.search_index]
        for search_value in uvn.get_values_by_type(search_type, case_index):

            # For each try to calculate the other arg
            for secondary_reverse_fn in self.secondary_reverse_fns:
                current_arg_list = []
                try:
                    other_value = secondary_reverse_fn(output_value, search_value)
                except Exception as e:
                    # TODO catch only hand thrown exceptions from dsl operators
                    # raise e
                    other_value = None
                    # continue
                if other_value is not None:
                    for arg_index, input_type in enumerate(self.operator.input_types):
                        value = search_value if arg_index == self.search_index else other_value
                        current_arg_list.append((input_type, value))
                    arg_lists.append(current_arg_list)

        return arg_lists


class TwoArgTrialSearch(ArgumentSearchStrategy):

    def __init__(self, operator: OperatorNode, validate_possible: Callable[[Type, Any], bool] = default_output_validation):
        super().__init__(operator, validate_possible)

    def _generate_argument_value_tuples_inner(self, uvn: UnifiedValueNetwork,
                                              output_type: Type, output_value: Any, case_index: int) -> List[List[Tuple[Type, Any]]]:
        arg_lists_expanded = []
        for arg_type in self.operator.input_types:
            arg_lists_expanded.append(uvn.get_values_by_type_no_outputs(arg_type, case_index))

        arg_lists = []
        for arg_combo in product(*arg_lists_expanded):

            try:
                if apply_operator_safe(self.operator, arg_combo) == output_value:
                    current_arg_list = []
                    for arg_index, arg_value in enumerate(arg_combo):
                        current_arg_list.append((self.operator.input_types[arg_index], arg_value))
                    arg_lists.append(current_arg_list)
            except Exception as e:
                # TODO catch only hand thrown exceptions from dsl operators
                # raise e
                continue
        return arg_lists


class SingleArgTrialSearch(ArgumentSearchStrategy):

    def __init__(self, operator: OperatorNode, validate_possible: Callable[[Type, Any], bool] = default_output_validation):
        super().__init__(operator, validate_possible)

    def _generate_argument_value_tuples_inner(self, uvn: UnifiedValueNetwork, output_type: Type, output_value: Any,
                                              case_index: int) -> List[List[Tuple[Type, Any]]]:

        arg_lists = []
        values = uvn.get_values_by_type_no_outputs(self.operator.input_types[0], case_index)
        print("SINGLE TRIAL. CASE:", case_index, "VALS:", len(values))
        for value in values:
            try:
                trial_val = apply_operator_safe(self.operator, [value])
                if trial_val == output_value:
                    arg_lists.append([(self.operator.input_types[0], value)])
            except Exception as e:
                print(e)
                # TODO catch only hand thrown exceptions from dsl operators
                # raise e
                continue

        return arg_lists

# Functions with two args where one of the args is injective from the output,
# and the other is to be derived from trial and error
class TwoArgPartialInjectiveTrialError(ArgumentSearchStrategy):
    ...


def ethans_function(oo: ObjectIdentity, oi: ObjectIdentity) -> Orientation:
    # TODO this is a placeholder, replace with actual function logic
    if oo.norm_image.array_hash != oi.norm_image.array_hash:
        raise ValueError("Input and output images do not match.")
    return difference_orientation(oi.orientation, oo.orientation)


OPERATOR_TO_STRATEGY: Dict[OperatorNode, ArgumentSearchStrategy] = {}

for op in OPERATOR_LIST:
    if isinstance(op, ApplyTwoDimensionalRelationshipOperator):
        OPERATOR_TO_STRATEGY[op] = TwoArgSearchAndCalculate(
            op, 1, [
                lambda o_o, i_o: None if o_o.x_location.env_shape != i_o.location.x_location.env_shape else MeasureCentersOperator.apply_with_second_arg_location(i_o, o_o),
                lambda o_o, i_o: None if o_o.x_location.env_shape != i_o.location.x_location.env_shape else MeasureClosestSidesOperator.apply_with_second_arg_location(i_o, o_o)
            ])

    if isinstance(op, MeasureCentersOperator):
        OPERATOR_TO_STRATEGY[op] = TwoArgTrialSearch(op)

    if isinstance(op, MeasureClosestSidesOperator):
        OPERATOR_TO_STRATEGY[op] = TwoArgTrialSearch(op)

    if isinstance(op, TranslateIdentityOperator):
        OPERATOR_TO_STRATEGY[op] = TwoArgSearchAndCalculate(
            op, 0, [
                ethans_function,
            ])

    if isinstance(op, ShapeSizeEnvOperator):
        OPERATOR_TO_STRATEGY[op] = SingleArgTrialSearch(op)

    # if isinstance(op, ApproachOperator):
    #     OPERATOR_TO_STRATEGY[op] = TwoArgTrialSearch(op)

    # if isinstance(op, ProportionalEnvOperator):
    #     OPERATOR_TO_STRATEGY[op] = TwoArgSearchAndCalculate(
    #         op,
    #         lambda output_value: [[(float, output_value / 2.0)], [(float, output_value * 2.0)]],
    #     )

# 2. Get operator semi-bijections if possible
# Case - 2d location relationship application
# - desired type is LocationValue, need ArcObject and relationship
# - By picking random arc objects (both i/o?) you can calculate the relationship


# Arc objects are a "measured" function only value. THIS IS OK, KEEP MOMENTUM


# NEED A TAXONOMY FOR THIS ^^. WHAT IS IT CALLED?
# There are infinite possible arc objects for any single relationship
# But there is only one relationship for every single arc object