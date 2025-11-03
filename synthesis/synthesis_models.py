import uuid
from abc import abstractmethod, ABC
from copy import copy
from enum import Enum
from typing import Optional, Tuple, Type, List, Any, Dict, TypeVar, Union, DefaultDict

from nodes import OperatorNode, Constant, Node, InputNode
from port_graphs import Edge
from operator_primitives import ConstructorOperator
from base_types import get_source_port_value
from synthesis.arc_specific.catalog_relationships_new import RelationshipSearchStrategy
from synthesis.program_state import TraceGraphInstance


class InstanceFormula:

    @abstractmethod
    def to_instance(self) -> TraceGraphInstance:
        pass


class InputInstanceFormula(InstanceFormula):

    def __init__(self, new_perceived_id: int, copied_instance: TraceGraphInstance):
        self.new_perceived_id = new_perceived_id
        self.copied_instance: TraceGraphInstance = copied_instance

    def to_instance(self, new_abstract_id: Optional[int]) -> TraceGraphInstance:
        self.copied_instance.put_abstract_concrete(new_abstract_id, self.new_perceived_id)
        return self.copied_instance


class OperatorInstanceFormula(InstanceFormula):

    def __init__(self, arg_tuples: List[Tuple[Type, Any]], copied_instance: TraceGraphInstance):
        self.arg_tuples = arg_tuples
        self.copied_instance: TraceGraphInstance = copied_instance

    def to_instance(self, operator_node_id: int, arg_ids: List[int]) -> TraceGraphInstance:

        if len(arg_ids) != len(self.arg_tuples):
            raise Exception("Bad assumption")

        for arg_index, arg_id in enumerate(arg_ids):
            self.copied_instance.state[arg_id] = self.arg_tuples[arg_index][1]
        self.copied_instance.fork_indexes[operator_node_id] = uuid.uuid4()
        return self.copied_instance


class ExtendUpEdgeFormula(InstanceFormula):

    def __init__(self, original_node_id: int, sub_path: Optional[str], copied_instance: TraceGraphInstance):
        self.original_node_id = original_node_id
        self.sub_path = sub_path
        self.copied_instance = copied_instance

    def to_instance(self, new_node_id: int) -> TraceGraphInstance:
        value = get_source_port_value(self.copied_instance.state[self.original_node_id], self.sub_path)
        self.copied_instance.state[new_node_id] = value
        return self.copied_instance


class ExtendUpNodeFormula(InstanceFormula):

    def __init__(self, port_to_arg_tuples: Dict[int, Tuple[Type, Any]], copied_instance: TraceGraphInstance):
        self.port_to_arg_tuples = port_to_arg_tuples
        self.copied_instance = copied_instance

    def to_instance(self, central_node_id: int, port_to_node_ids: Dict[int, int]) -> TraceGraphInstance:

        if len(port_to_node_ids) != len(self.port_to_arg_tuples):
            raise Exception("Bad assumption")
        self.copied_instance.state = copy(self.copied_instance.state)
        self.copied_instance.state[central_node_id] = True

        for port_index, (arg_type, arg_value) in self.port_to_arg_tuples.items():
            self.copied_instance.state[port_to_node_ids[port_index]] = arg_value

        return self.copied_instance

class BasicSynthesisActions(Enum):
    EXTEND_DOWN = 0
    EXTEND_UP = 1
    MERGE_DOWN = 2
    MERGE_UP = 3
    REFOCUS = 4

class DirectiveType(Enum):
    """Enum for all directive types including granular subdirectives"""
    EXTEND_DOWN_CONSTANT = "extend_down_constant"
    EXTEND_DOWN_INPUT = "extend_down_input"
    EXTEND_DOWN_CONSTRUCTOR = "extend_down_constructor"
    EXTEND_DOWN_OPERATOR = "extend_down_operator"
    EXTEND_UP_EDGE = "extend_up_edge"
    EXTEND_UP_NODE = "extend_up_node"
    FORK_JOIN = "fork_join"
    LEAF_JOIN = "leaf_join"
    GENERIC_MERGE = "generic_merge"
    REFOCUS_PREDICATE = "refocus_predicate"


DIRECTIVE_TYPE_TO_BASIC_ACTION: Dict[DirectiveType, BasicSynthesisActions] = {
    DirectiveType.EXTEND_DOWN_CONSTANT: BasicSynthesisActions.EXTEND_DOWN,
    DirectiveType.EXTEND_DOWN_INPUT: BasicSynthesisActions.EXTEND_DOWN,
    DirectiveType.EXTEND_DOWN_CONSTRUCTOR: BasicSynthesisActions.EXTEND_DOWN,
    DirectiveType.EXTEND_DOWN_OPERATOR: BasicSynthesisActions.EXTEND_DOWN,
    DirectiveType.EXTEND_UP_EDGE: BasicSynthesisActions.EXTEND_UP,
    DirectiveType.EXTEND_UP_NODE: BasicSynthesisActions.EXTEND_UP,
    DirectiveType.FORK_JOIN: BasicSynthesisActions.MERGE_DOWN,
    DirectiveType.LEAF_JOIN: BasicSynthesisActions.MERGE_UP,
    DirectiveType.REFOCUS_PREDICATE: BasicSynthesisActions.REFOCUS,
    # GENERIC_MERGE = "generic_merge"
}


class DirectiveKey(ABC):
    def __init__(self, directive_type: DirectiveType):
        self.directive_type = directive_type

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        pass


INST_KEY = TypeVar('INST_KEY', bound=DirectiveKey)
INST_FORMULA = TypeVar('INST_FORMULA', bound=Union[TraceGraphInstance,InstanceFormula])


class TraceInstanceMapping(DefaultDict[INST_KEY, Dict[int, List[INST_FORMULA]]]):
    def __init__(self, **kwargs) -> None:
        super().__init__(dict, **kwargs)


class ExtendDownConstantKey(DirectiveKey):
    def __init__(self, value_type: Type, value: Any):
        super().__init__(DirectiveType.EXTEND_DOWN_CONSTANT)
        self.value_type = value_type
        self.value = value

    def __hash__(self) -> int:
        return hash((self.directive_type, self.value_type, self.value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtendDownConstantKey):
            return False
        return self.value_type == other.value_type and self.value == other.value

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return Constant


class ExtendDownInputKey(DirectiveKey):
    def __init__(self, input_type: Type, existing_node_id: Optional[int], path: Optional[str]):
        super().__init__(DirectiveType.EXTEND_DOWN_INPUT)
        self.input_type = input_type
        self.existing_node_id = existing_node_id
        self.path = path

    def __hash__(self) -> int:
        return hash((self.directive_type, self.input_type, self.existing_node_id, self.path))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtendDownInputKey):
            return False
        return (self.input_type == other.input_type and
                self.existing_node_id == other.existing_node_id and
                self.path == other.path)

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return InputNode


class ExtendDownConstructorKey(DirectiveKey):
    def __init__(self, constructor_index: int):
        super().__init__(DirectiveType.EXTEND_DOWN_CONSTRUCTOR)
        self.constructor_index = constructor_index

    def __hash__(self) -> int:
        return hash((self.directive_type, self.constructor_index))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtendDownConstructorKey):
            return False
        return self.constructor_index == other.constructor_index

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return ConstructorOperator


class ExtendDownOperatorKey(DirectiveKey):
    def __init__(self, operator: OperatorNode):
        super().__init__(DirectiveType.EXTEND_DOWN_OPERATOR)
        self.operator = operator

    def __hash__(self) -> int:
        return hash((self.directive_type, self.operator))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtendDownOperatorKey):
            return False
        return self.operator == other.operator

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return self.operator.__class__


class ExtendUpEdgeKey(DirectiveKey):
    def __init__(self, sub_path: Optional[str], sub_type: Type):
        super().__init__(DirectiveType.EXTEND_UP_EDGE)
        self.sub_path = sub_path
        self.sub_type = sub_type

    def __hash__(self) -> int:
        return hash((self.directive_type, self.sub_path, self.sub_type))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtendUpEdgeKey):
            return False
        return self.sub_path == other.sub_path and self.sub_type == other.sub_type

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return Edge


class ExtendUpNodeKey(DirectiveKey):
    def __init__(self, relationship_strategy: RelationshipSearchStrategy):
        super().__init__(DirectiveType.EXTEND_UP_NODE)
        self.relationship_strategy = relationship_strategy

    def __hash__(self) -> int:
        return hash((self.directive_type, self.relationship_strategy))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExtendUpNodeKey):
            return False
        return self.relationship_strategy == other.relationship_strategy

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return self.relationship_strategy.relationship.__class__


class ForkJoinKey(DirectiveKey):
    def __init__(self, duplicate_abstract_nodes: Tuple[Tuple]):
        super().__init__(DirectiveType.FORK_JOIN)
        self.duplicate_abstract_nodes = duplicate_abstract_nodes

    def __hash__(self) -> int:
        return hash((self.directive_type, self.duplicate_abstract_nodes))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ForkJoinKey):
            return False
        return self.duplicate_abstract_nodes == other.duplicate_abstract_nodes

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return None


class LeafJoinKey(DirectiveKey):
    def __init__(self, duplicate_abstract_nodes: Tuple[Tuple]):
        super().__init__(DirectiveType.LEAF_JOIN)
        self.duplicate_abstract_nodes = duplicate_abstract_nodes

    def __hash__(self) -> int:
        return hash((self.directive_type, self.duplicate_abstract_nodes))

    def __eq__(self, other) -> bool:
        if not isinstance(other, LeafJoinKey):
            return False
        return self.duplicate_abstract_nodes == other.duplicate_abstract_nodes

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return None


class GenericMergeKey(DirectiveKey):
    def __init__(self, dfs_code_str: str):
        super().__init__(DirectiveType.GENERIC_MERGE)
        self.dfs_code_str = dfs_code_str

    def __hash__(self) -> int:
        return hash((self.directive_type, self.dfs_code_str))

    def __eq__(self, other) -> bool:
        if not isinstance(other, GenericMergeKey):
            return False
        return self.dfs_code_str == other.dfs_code_str

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return None


class RefocusPredicateKey(DirectiveKey):
    def __init__(self, new_focus_node_id: int):
        super().__init__(DirectiveType.REFOCUS_PREDICATE)
        self.new_focus_node_id = new_focus_node_id

    def __hash__(self) -> int:
        return hash((self.directive_type, self.new_focus_node_id))

    def __eq__(self, other) -> bool:
        if not isinstance(other, RefocusPredicateKey):
            return False
        return self.new_focus_node_id == other.new_focus_node_id

    def get_dsl_resource(self) -> Optional[Union[Type[Node], Type[Edge]]]:
        return None
