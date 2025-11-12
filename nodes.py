from copy import copy
from typing import List, Any, Tuple, Iterable, Type, Optional, Set

from base_types import DslSet

DEFAULT_GROUP_BY = "default"


class Node:
    __global_id = 0

    @staticmethod
    def get_next_id():
        Node.__global_id += 1
        return Node.__global_id

    def __init__(self, input_port_count):
        self.id = Node.get_next_id()
        self.input_port_count = input_port_count

    def get_debug_label(self):
        return None

    def get_outbound_type(self) -> Type:
        ...

    # Creates a copy, assigns a new ID, and returns a tuple of the new node and the OLD node id
    def copy_and_refresh_id(self) -> Tuple['Node', int]:
        new_node = copy(self)

        # NOTE MANUALLY ASSIGNING NEW ID
        original_id = new_node.id
        new_node.id = Node.get_next_id()
        return new_node, original_id


class Constant(Node):

    def __init__(self, value_type: Type, value):
        super().__init__(0)
        self.value_type = value_type
        self.value = value

    def get_debug_label(self):
        return f"{str(self.value_type)} - {self.value}"

    def get_outbound_type(self) -> Type:
        return self.value_type


class InputNode(Node):

    def __init__(self, value_type: Type, perception_qualifiers: Optional[Set[Any]]):
        super().__init__(0)
        self.value_type = value_type
        self.perception_qualifiers = perception_qualifiers if isinstance(perception_qualifiers, set) else ({
            perception_qualifiers} if perception_qualifiers is not None else set())

    def get_debug_label(self):
        return " ".join((p if isinstance(p, str) else p.__name__) for p in self.perception_qualifiers)

    def get_outbound_type(self):
        return self.value_type


class InputSetNode(Node):

    def __init__(self, member_type: Type, perception_qualifiers: Optional[Set[Any]] = None, group_by = None):
        super().__init__(0)
        self.member_type = member_type
        self.perception_qualifiers = perception_qualifiers
        self.group_by = group_by

    def get_debug_label(self):
        return f"{self.member_type.__name__} set"

    def get_outbound_type(self) -> Type:
        return DslSet


class OutputNode(Node):

    def __init__(self, value_type):
        super().__init__(1)
        self.value_type = value_type

    def get_debug_label(self):
        return self.__class__.__name__

    def get_outbound_type(self):
        return self.value_type


class RelationshipNode(Node):

    def __init__(self, input_types: List):
        super().__init__(len(input_types))
        self.input_types = input_types

    def apply(self, *args) -> bool:
        ...

    # NOTE need to upgrade this signature for relationships that have > 2 input ports and
    #  subsets of symmetric ports (like ARC's between cardinal)
    def is_symmetrical(self) -> bool:
        return False

    def get_outbound_type(self) -> Type:
        return bool


class OperatorNode(Node):

    def __init__(self, input_types: List, output_type):
        super().__init__(len(input_types))
        self.input_types = input_types
        self.output_type = output_type

    def apply(self, *args):
        ...

    def primitive(self) -> bool:
        return True

    def get_outbound_type(self) -> Type:
        return self.output_type


### Proxy (looping) nodes ###


class ProxyNode(Node):
    def __init__(self):
        super().__init__(2)

    def apply(self, initial_value: Any, continuous_value: Any):
        raise Exception("proxy is not applied as a normal node. Requires special interpreter handling")

    # NOTE: Once capable of recursive program synthesis, it will be important to infer this from its args,
    # which also should enforce they are the same type.
    def get_outbound_type(self) -> Type:
        return Any


# Takes a scalar, create a copy of the graph instance using this scalar as the node's value
# Graph instance is appended to the end of the graph instance queue
class RecursiveProxyNode(ProxyNode):
    pass


# Takes a set, creates a copy of the graph instance for each scalar in the set using it as the node's value
# Graph instances are appended to the start of the graph instance queue
class RecursiveSearchProxyNode(ProxyNode):
    pass


# Takes a scalar and adds it as this nodes value in the next graph instance (ordered)
class IterativeProxyNode(ProxyNode):
    pass


class SearchConditionNode(Node):

    def __init__(self, condition: RelationshipNode):
        super().__init__(condition.input_port_count)
        self.condition = condition

    def apply(self, *args) -> bool:
        return self.condition.apply(*args)


### SET OPERATION NODES ###

# Takes one argument: the thing to add to the set. Should run all graph instances up to this node, then put the same
# set in every instance
class SetJoin(Node):
    def __init__(self):
        super().__init__(1)

    @staticmethod
    def apply(single_value: Any):
        return single_value

    def get_outbound_type(self) -> Type:
        return DslSet


class SetSplit(Node):

    # NOTE type should really be inferred based on the input set's member type.
    def __init__(self, inner_type):
        super().__init__(1)
        self.inner_type = inner_type

    @staticmethod
    def apply(set_value: Iterable):
        if not isinstance(set_value, DslSet):
            raise TypeError("Unexpected set node type: " + str(type(set_value)))
        return set_value

    def get_outbound_type(self) -> Type:
        return self.inner_type


# Node which collects values to a set, and shares them with another graph,
# but in a way allowing that this connection does NOT count as a connection of the graphs in the interpreter.
# Meaning if it is the only common node between the two subgraphs, they will be evaluated as disjoint graphs with
# order enforced.
class DisjointSetNode(Node):

    def __init__(self, member_type: Type):
        super().__init__(1)
        self.member_type = member_type

    def get_debug_label(self):
        return self.__class__.__name__

    def get_outbound_type(self) -> Type:
        return DslSet[self.member_type]


### MISC NODES ###

# Used to control the order of evaluation of graph instances during the computation stage
# Must be in the population stage subgraph, while that is still a concept.
class RuntimeOrderNode(Node):

    def __init__(self):
        super().__init__(1)

    @staticmethod
    def apply(comparable: int) -> int:
        if not isinstance(comparable, int):
            raise Exception("RuntimeOrderNode only supports int types")
        return comparable

    def get_outbound_type(self) -> Type:
        return int


### GENERIC NODES ARE FOR INTERNAL USE ONLY ###

class GenericBaseNode:
    ...


class GenericConstantNode(Constant, GenericBaseNode):

    def __init__(self):
        super().__init__(Any, None)


class GenericRelationshipNode(RelationshipNode, GenericBaseNode):

    def __init__(self):
        super().__init__([])


class GenericOperatorNode(OperatorNode, GenericBaseNode):

    def __init__(self):
        super().__init__([],Any)


