from collections import defaultdict
from typing import List, Type, DefaultDict, Tuple, Set

from nodes import Node, InputNode, InputSetNode, OutputNode, RecursiveProxyNode, SetJoin, SetSplit
from operator_primitives import AdditionOperator, SubtractionOperator, MultiplyOperator, \
    SumSetOperator, SetRankOperator, ApplyScalarOpToSetOperator, CreateLocalSetOperator, AddToLocalSetOperator, \
    ConstructorOperator
from relationship_primitives import Equals, NotEquals, LessThan, LessThanOrEqual, GreaterThan, \
    GreaterThanOrEqual, SetContains, SetNotContains
from base_types import DslType, PerceivedType


class Library:

    def __init__(self):
        self.__types: Set[Type] = set()
        self.__nodes: Set[Type[Node]] = set()
        self.__perceived_types: Set[Type[PerceivedType]] = set()

        # Type to list of tuples (path, value type) of nested types (flattened, like "child.grandchild...")
        self.__type_to_nested_type_flattened: DefaultDict[Type[DslType], Set[Tuple[str, Type]]] = defaultdict(set)

        # Reverse lookup of the __type_to_nested_type_flattened
        self.__type_to_parent_type_flattened: DefaultDict[Type, Set[Tuple[str, Type[DslType]]]] = defaultdict(set)

    def add_value_type(self, value_type: Type):
        if issubclass(value_type, DslType):
            if issubclass(value_type, PerceivedType):
                self.__perceived_types.add(value_type)
            self.__traverse_and_record_type(value_type)
        self.__types.add(value_type)

    def add_node_type(self, node_type: Type[Node]):
        self.__nodes.add(node_type)

    # traverse the type recursively, record both sides
    def __traverse_and_record_type(self, dsl_type: Type[DslType]):

        # Start with default empty path, traverse down
        queue = [(None, dsl_type)]
        while queue:
            # NOTE - missing infinite loop safety
            current_path, current_type = queue.pop(0)

            self.__type_to_nested_type_flattened[dsl_type].add((current_path, current_type))
            if current_type != dsl_type:
                self.__type_to_parent_type_flattened[current_type].add((current_path, dsl_type))

            if issubclass(current_type, DslType):
                for component_path, component_type in current_type.get_components():
                    path_base = "" if current_path is None else current_path + "."
                    full_path = path_base + component_path
                    queue.append((full_path, component_type))

    def get_flattened_components(self, parent_type: Type) -> List[Tuple[str, Type]]:
        return self.__type_to_nested_type_flattened[parent_type]


### Singleton Default Library Instance

DEFAULT_LIBRARY = Library()

### NODES

# 'Keyword' Nodes
DEFAULT_LIBRARY.add_node_type(InputNode)
DEFAULT_LIBRARY.add_node_type(InputSetNode)
DEFAULT_LIBRARY.add_node_type(OutputNode)
DEFAULT_LIBRARY.add_node_type(RecursiveProxyNode)
DEFAULT_LIBRARY.add_node_type(SetJoin)
DEFAULT_LIBRARY.add_node_type(SetSplit)

# Constructor
DEFAULT_LIBRARY.add_node_type(ConstructorOperator)

# Arithmetic Operators
DEFAULT_LIBRARY.add_node_type(AdditionOperator)
DEFAULT_LIBRARY.add_node_type(SubtractionOperator)
DEFAULT_LIBRARY.add_node_type(MultiplyOperator)

# Set Operators
DEFAULT_LIBRARY.add_node_type(SumSetOperator)
DEFAULT_LIBRARY.add_node_type(SetRankOperator)
DEFAULT_LIBRARY.add_node_type(ApplyScalarOpToSetOperator)
DEFAULT_LIBRARY.add_node_type(CreateLocalSetOperator)
DEFAULT_LIBRARY.add_node_type(AddToLocalSetOperator)

# Comparison Relationships
DEFAULT_LIBRARY.add_node_type(Equals)
DEFAULT_LIBRARY.add_node_type(NotEquals)
DEFAULT_LIBRARY.add_node_type(LessThan)
DEFAULT_LIBRARY.add_node_type(LessThanOrEqual)
DEFAULT_LIBRARY.add_node_type(GreaterThan)
DEFAULT_LIBRARY.add_node_type(GreaterThanOrEqual)

# Set Relationships
DEFAULT_LIBRARY.add_node_type(SetContains)
DEFAULT_LIBRARY.add_node_type(SetNotContains)


### Types
# All default types inherited from python
