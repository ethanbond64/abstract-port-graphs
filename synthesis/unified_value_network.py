from enum import Enum
from collections import defaultdict
from copy import copy
from itertools import product
from time import time
from typing import Type, Any, Optional, Callable, Dict, List, Tuple, DefaultDict, Set

from nodes import OperatorNode, Constant, Node
from port_graphs import PortGraph
from operator_primitives import ConstructorOperator
from base_types import get_source_port_value
from synthesis.design_time_models import PerceivedValue

CACHING_ENABLED = True


class UVNNode:
    __global_id = 0

    @staticmethod
    def get_next_id():
        UVNNode.__global_id += 1
        return UVNNode.__global_id

    def __init__(self):
        self.id = UVNNode.get_next_id()
        self.created_on = time()


class ValueNode(UVNNode):

    def __init__(self, value_type: Type, value: Any):
        UVNNode.__init__(self)
        self.value_type = value_type
        self.value = value

        # TODO temp until I figure out where to cache these, WITH A CURSOR
        # case index -> Tuple [ List of discovered path nodes, Set of omitted value nodes ]
        self.temp_loaded_paths_by_case: Dict[int, Tuple[List[PathNode], Set[int]]] = {}

    def __eq__(self, __value):

        if __value is None:
            return False

        if not isinstance(__value, ValueNode):
            return False

        return self.value_type == __value.value_type and self.value == __value.value # TODO temp BAD HASH IN EQUALS IS UNSAFE

    def __hash__(self):
        return hash((self.value_type, self.value))


class SourceType(Enum):
    CONSTANT = 0
    INPUT = 1
    ATTRIBUTE = 2
    CONSTRUCTOR = 3
    OPERATION = 4
    OTHER_OUTPUT = 5
    WILDCARD = 6


class SourceNode(UVNNode):

    def __init__(self, value_id: int, value_type: Type):
        UVNNode.__init__(self)
        self.value_id = value_id
        self.value_type = value_type

    def is_terminal(self):
        return False

    def get_dependent_values(self) -> Tuple[int]:
        return tuple()

    def get_identifiable_values(self) -> Any:
        return None

    def get_source_type(self):
        ...

    def __eq__(self, other):
        if other is None:
            return False
        return (self.value_id == other.value_id and self.get_source_type() == other.get_source_type() and
                self.get_dependent_values() == other.get_dependent_values() and
                self.get_identifiable_values() == other.get_identifiable_values())

    def __hash__(self):
        return hash((self.value_id, self.get_source_type(),
                     self.get_dependent_values(), self.get_identifiable_values()))

class TerminalSourceNode(SourceNode):

    def __init__(self, value_id: int, value_type: Type, constant: bool,
                 perceived_value_id: Optional[int] = None, case_index: Optional[int] = None):
        super().__init__(value_id, value_type)
        self.constant = constant
        self.perceived_value_id = perceived_value_id
        self.case_index = case_index

    def is_terminal(self):
        return True

    def get_source_type(self):
        return SourceType.CONSTANT if self.constant else SourceType.INPUT

    def get_identifiable_values(self) -> Any:
        return self.perceived_value_id, self.case_index


class AttributeSourceNode(SourceNode):

    def __init__(self, value_id: int, value_type: Type, attribute_path: str, source_value: int):
        super().__init__(value_id, value_type)
        self.attribute_path = attribute_path
        self.source_value = source_value

    def get_dependent_values(self) -> Tuple[int]:
        return (self.source_value,)

    def get_source_type(self):
        return SourceType.ATTRIBUTE

    def get_identifiable_values(self) -> Any:
        return self.attribute_path


class ConstructorSourceNode(SourceNode):

    def __init__(self, value_id: int, value_type: Type, constructor: Callable, source_values: Tuple[int]):
        super().__init__(value_id, value_type)

        self.constructor = constructor
        self.source_values = source_values

    def get_dependent_values(self) -> Tuple[int]:
        return self.source_values

    def get_source_type(self):
        return SourceType.CONSTRUCTOR

    def get_identifiable_values(self) -> Any:
        # NOTE IF MULTIPLE CONSTRUCTORS EVER ALLOWED THIS NEEDS TO CHANGE
        return self.value_type


class OperationalSourceNode(SourceNode):

    def __init__(self, value_id: int, value_type: Type, operator: Node, source_values: Tuple[int]):
        super().__init__(value_id, value_type)

        if operator.get_outbound_type() != value_type:
            raise Exception()

        self.operator = operator.__class__
        self.input_types = operator.input_types if isinstance(operator, OperatorNode) else [] # TODO invalid
        self.source_values = source_values

    def get_dependent_values(self) -> Tuple[int]:
        return self.source_values

    def get_source_type(self):
        return SourceType.OPERATION

    def get_identifiable_values(self) -> Any:
        return self.operator


# Recursive nodes don't need a tag/source for now. They are just a setting on the search,
# allowing it to treat output values as if they are accessible to input values.
# class RecursiveSourceNode(SourceNode):
#
#     def __init__(self, value_id: int, value_type: Type, case_index: int):
#         super().__init__(value_id, value_type, case_index)

class OutputTagNode(UVNNode):

    def __init__(self, value_id: int, value_type: Type, case_index: int,
                 perceived_value_id: int):
        super().__init__()
        self.value_id = value_id
        self.value_type = value_type
        self.case_index = case_index
        self.perceived_value_id = perceived_value_id


class PathNode:

    def __init__(self, value_id: int, source_id: int, source_type: SourceType, children: List['PathNode']):
        self.value_id = value_id
        self.source_id = source_id
        self.source_type = source_type
        self.children = children

        self.cumulative_inputs: Set[int] = set()
        self.cumulative_constants: Set[int] = set()

        self.__lazy_inorder_tuple = None

        if source_type == SourceType.INPUT:
            self.cumulative_inputs.add(value_id)
        elif source_type == SourceType.CONSTANT:
            self.cumulative_constants.add(value_id)
        else:
            self.cumulative_inputs = self.cumulative_inputs.union(*list(map(lambda c: c.cumulative_inputs, children)))
            self.cumulative_constants = self.cumulative_constants.union(*list(map(lambda c: c.cumulative_constants, children)))

    def to_inorder_tuple(self) -> Tuple[int, Tuple[int]]:
        if self.__lazy_inorder_tuple is None:
            self.__lazy_inorder_tuple = self.source_id, tuple(child.to_inorder_tuple() for child in self.children)
        return self.__lazy_inorder_tuple

    def get_depth(self):
        return (0 if self.source_type == SourceType.ATTRIBUTE else 1) + (max(child.get_depth() for child in self.children) if self.children else 0)

    def get_size(self):
        return 1 + sum(child.get_size() for child in self.children)

    def __eq__(self, __value):
        if __value is None or not isinstance(__value, PathNode):
            return False
        return self.to_inorder_tuple() == __value.to_inorder_tuple()

    def __hash__(self):
        return hash(self.to_inorder_tuple())

class UnifiedValueNetwork:

    def __init__(self, class_component_catalog: Dict[Type, Dict[str, Type]]):
        # self.class_constructor_catalog = class_constructor_catalog
        self.class_component_catalog = class_component_catalog

        # Value nodes
        self.value_nodes_by_id: Dict[int, ValueNode] = {}
        self.value_nodes_to_id: Dict[ValueNode, int] = {}

        # Source nodes
        self.source_nodes_by_id: Dict[int, SourceNode] = {}
        self.source_nodes_to_id: Dict[SourceNode, int] = {}

        self.source_nodes_by_value_id_case_agnostic: DefaultDict[int, Set[SourceNode]] = defaultdict(set)
        self.source_nodes_by_value_id_and_case: DefaultDict[Tuple[int, int], Set[SourceNode]] = defaultdict(set)

        # Attribute specific source node lookup (value node id, to set of attribute source nodes which it provides
        self.attribute_reverse_source_nodes_by_value_id: DefaultDict[int, Set[AttributeSourceNode]] = defaultdict(set)

        # Output tags
        self.output_tags_to_value_id: Dict[OutputTagNode, int] = {}
        self.output_tags_by_value_id: DefaultDict[int, Set[OutputTagNode]] = defaultdict(set)
        self.output_values_to_cases: DefaultDict[int, Set[int]] = defaultdict(set)
        self.cases_to_output_values: DefaultDict[int, Set[int]] = defaultdict(set)

        # Perception storage
        # Keyed by tuple (perception_model id (int), PerceivedValueType (type), output_side (bool))
        self.perceived_values_by_model_and_type: DefaultDict[Tuple[int, Type, bool], List[PerceivedValue]] = defaultdict(list)

    # NULLABLE if not an input
    def get_input_external_ids(self, value_type: Type, value: any, case_index: int) -> Set[int]:
        value_node = self.__lazy_create_value_node(value_type, value)
        source_nodes = self.source_nodes_by_value_id_and_case[(value_node.id, case_index)]
        return set(map(lambda sn: sn.perceived_value_id,
                   filter(lambda sn: sn.get_source_type() == SourceType.INPUT and
                                     sn.case_index == case_index, source_nodes)))

    def get_all_inputs_by_ids(self, value_type: Type) -> Dict[int, Any]:
        return {sn.perceived_value_id: self.value_nodes_by_id[sn.value_id].value for sn in
                filter(lambda sn: sn.value_type == value_type and sn.get_source_type() == SourceType.INPUT, self.source_nodes_by_id.values())}

    def get_output_external_ids(self, value_type: Type, value: any, case_index: int) -> Set[int]:
        value_node = self.__lazy_create_value_node(value_type, value)
        return set(map(lambda ot: ot.perceived_value_id, filter(lambda ot: ot.case_index == case_index, self.output_tags_by_value_id[value_node.id])))

    def is_output_value(self, value_type: Type, value: any, case_index: int):
        return len(self.get_output_external_ids(value_type, value, case_index)) > 0

    def get_input_attribute_paths(self, value_type: Type, value: any, case_index: int) -> Set[Tuple[Type, int, Optional[str]]]:
        value_node = self.__lazy_create_value_node(value_type, value)
        output: Set[Tuple[int, str]] = set()

        def path_consumer(last_node_id: int, path: List[str], case_idx=case_index, out=output):
            concat_path = ".".join(reversed(path)) if len(path) > 0 else None
            for sn in self.source_nodes_by_value_id_and_case[(last_node_id, case_idx)]:
                if sn.get_source_type() == SourceType.INPUT:
                    out.add((sn.value_type, sn.perceived_value_id, concat_path))

        self.search_paths_with_two_pole_grammar_IMPL(value_node.id, path_consumer)
        return output

    # Get all values of a given type in the UVN that are NOT output values themselves
    # If they are only sourced by inputs, make sure at least of of the input sources has the specific case index
    def get_values_by_type_no_outputs(self, value_type: Type, case_index: int) -> Set[Any]:
        value_nodes = set(filter(lambda vn: vn.value_type == value_type, self.value_nodes_by_id.values()))

        blacklist = self.create_input_based_blacklist(case_index, value_nodes, True)

        return set(vn.value for vn in value_nodes if vn.id not in blacklist)

    def create_input_based_blacklist(self, case_index, value_nodes, filter_exclusive_outputs=False):
        # if there are any non-case agnostic source of this value id, AND NONE of them are the arg case index, skip the value
        # This is because this method is used for value search in operators, and we don't want to waste time trial/erroring
        # perceivable values not rooted in the case (for now)
        value_ids_and_cases = {k for vn in value_nodes
                               for k, v in self.source_nodes_by_value_id_and_case.items()
                               if k[0] == vn.id and len(v) > 0}
        values_to_cases = defaultdict(set)
        for value_id, case in value_ids_and_cases:
            values_to_cases[value_id].add(case)
        blacklist = set()
        for value, cases in values_to_cases.items():
            if case_index not in cases:
                blacklist.add(value)

        # Outputs that are ONLY outputs will not show up in the case lookup.
        if filter_exclusive_outputs:
            for og_vn in value_nodes:
                if og_vn.id not in values_to_cases and len(self.output_tags_by_value_id[og_vn.id]) > 0:
                    blacklist.add(og_vn.id)
        return blacklist

    def get_values_by_type(self, value_type: Type, case_index: int) -> Set[Any]:
        value_nodes = set(filter(lambda vn: vn.value_type == value_type, self.value_nodes_by_id.values()))
        input_blacklist = self.create_input_based_blacklist(case_index, value_nodes)

        output_blacklist = set(vn.id for vn in value_nodes
                               if len(self.output_tags_by_value_id[vn.id]) > 0 and
                               all(tag.case_index != case_index for tag in self.output_tags_by_value_id[vn.id]))

        return set(vn.value for vn in value_nodes if vn.id not in input_blacklist and vn.id not in output_blacklist)

    def TEMP_get_value_node(self, value_type: Type, value: Any) -> ValueNode:
        return self.__lazy_create_value_node(value_type, value)


    def add_constant_value(self, value_type: Type, value: Any):
        value_node = self.__lazy_create_value_node(value_type, value)
        constant_source = TerminalSourceNode(value_node.id, value_type, True, None)
        self.__register_source(constant_source, None)

    def add_perceived_value(self, perceived_value: PerceivedValue):
        if not perceived_value.test:
            perception_key = (perceived_value.perception_model_id, perceived_value.value_type, perceived_value.output)
            self.perceived_values_by_model_and_type[perception_key].append(perceived_value)
            if perceived_value.output:
                self.add_output_value(perceived_value.value_type, perceived_value.value,
                                      perceived_value.case_index, perceived_value.id)
            else:
                self.add_input_value(perceived_value.value_type, perceived_value.value,
                                        perceived_value.case_index, perceived_value.id)

    def get_perceived_values(self, perception_model_id: int, value_type: Type, output: bool) -> List[PerceivedValue]:
        key = (perception_model_id, value_type, output)
        return self.perceived_values_by_model_and_type[key]

    def add_input_value(self, value_type: Type, value: Any, case_index: int, perceived_value_id: int):
        value_node = self.__lazy_create_value_node(value_type, value)
        input_source = TerminalSourceNode(value_node.id, value_type, False, perceived_value_id, case_index)
        self.__register_source(input_source, case_index)
        self.__add_value_and_child_attributes(value_node, value_type, value)

    def add_output_value(self, value_type: Type, value: Any, case_index: int, perceived_value_id: int):
        value_node = self.__lazy_create_value_node(value_type, value)
        # TODO output source to come later
        self.__add_value_and_child_attributes(value_node, value_type, value)

        output_tag = OutputTagNode(value_node.id, value_type, case_index, perceived_value_id)
        self.output_tags_by_value_id[value_node.id].add(output_tag)
        self.output_tags_to_value_id[output_tag] = value_node.id
        self.output_values_to_cases[value_node.id].add(case_index)
        self.cases_to_output_values[case_index].add(value_node.id)


    def __add_value_and_child_attributes(self, value_node: ValueNode, value_type: Type, value: Any):
        # TODO duplicate recursive component registration from where the constants come from in the programming system
        value_queue: List[Tuple[str, Type, Any]] = [(value_node, value_type, value)]
        while value_queue:
            current_tuple = value_queue.pop(0)
            for attr_path, attr_type in self.class_component_catalog.get(current_tuple[1], {}).items():
                attr_value = get_source_port_value(current_tuple[2], attr_path)
                # if attr_value is not None:
                attr_value_node = self.__add_attribute_value(current_tuple[0], attr_type, attr_value, attr_path)
                value_queue.append((attr_value_node, attr_type, attr_value))

    def add_attribute_value(self, value_type: Type, value: Any, target_type: Type, target_value: Any, path: str):
        value_node = self.__lazy_create_value_node(value_type, value)
        self.__add_attribute_value(value_node, target_type, target_value, path)

    def add_operation_source(self, operator_node: Node, value_type: Type, value: Any,
                             inputs: List[Tuple[Type, Any]], case_index: int):
        value_node = self.__lazy_create_value_node(value_type, value)

        input_nodes = []
        for input_type, input_value in inputs:
            input_node = self.__lazy_create_value_node(input_type, input_value)
            input_nodes.append(input_node.id)

        operator_source = OperationalSourceNode(value_node.id, value_type, operator_node, tuple(input_nodes))
        self.__register_source(operator_source, None)


    ### UVN TRAVERSAL METHODS ###

    def get_declared_constants(self, value_type: Optional[Type] = None) -> List[ValueNode]:
        return list(
            map(lambda sn: self.value_nodes_by_id[sn.value_id],
                filter(lambda sn: sn.get_source_type() == SourceType.CONSTANT and
                                  (value_type is None or sn.value_type == value_type), self.source_nodes_to_id.keys())))

    # Returns list of tuples: (Value type, Value, external id, case index)
    def get_output_values(self) -> List[Tuple[Type, Any, int, int]]:
        tuples = []
        for value_id, output_tags in self.output_tags_by_value_id.items():
            value_node = self.value_nodes_by_id[value_id]
            for output_tag in output_tags:
                tuples.append((value_node.value_type, value_node.value, output_tag.perceived_value_id, output_tag.case_index))

        return tuples

    def search_paths_with_two_pole_grammar_IMPL(self, start_node_id: int, path_consumer: Callable):

        queue: List[Tuple[int, List[str]]] = [(start_node_id, [])]

        while queue:
            current_node_id, current_path = queue.pop(0)

            # Look for nodes this node is an attribute of. enqueue
            for attribute_node in filter(lambda sn: sn.get_source_type() == SourceType.ATTRIBUTE and
                                                    sn.value_id == current_node_id, self.source_nodes_to_id.keys()):
                for parent_value_id in attribute_node.get_dependent_values():
                    next_path = list(current_path)
                    next_path.append(attribute_node.attribute_path)
                    queue.append((parent_value_id, next_path))

            # pass to consumer - which will determine if this is a valid "terminal"
            # node in this grammar and record it.
            path_consumer(current_node_id, current_path)

    def get_terminal_source_paths_by_value(self, value: Any, value_type: Type, case_index: int) -> List[PathNode]:
        value_node = self.__lazy_create_value_node(value_type, value)
        return self.get_terminal_source_paths(value_node, case_index)

    def get_terminal_source_paths(self, value_node: ValueNode, case_index: int,
                                  context_nodes: Optional[Set[int]] = None) -> List[PathNode]:
        seen_nodes = set(context_nodes) if context_nodes is not None else set()
        seen_nodes.add(value_node.id)

        cached_paths = None
        exclusive_nodes = None
        omitted_nodes = set()
        if CACHING_ENABLED:
            if case_index in value_node.temp_loaded_paths_by_case.keys():
                cached_paths, omitted_nodes = value_node.temp_loaded_paths_by_case[case_index]
                exclusive_nodes = omitted_nodes.difference(seen_nodes)
                if len(exclusive_nodes) == 0:
                    return cached_paths

        source_paths: List[PathNode] = []

        all_source_nodes = set()
        all_source_nodes = all_source_nodes.union(self.source_nodes_by_value_id_and_case[(value_node.id, case_index)])
        all_source_nodes = all_source_nodes.union(self.source_nodes_by_value_id_case_agnostic[value_node.id])

        for source_node in sorted(all_source_nodes, key=lambda n: n.id):

            if source_node.is_terminal():
                path_node: PathNode = PathNode(value_node.id, source_node.id, source_node.get_source_type(), [])
                source_paths.append(path_node)
            else:
                child_paths: List[List[PathNode]] = []
                for dependent_value in source_node.get_dependent_values():
                    if dependent_value not in seen_nodes and (exclusive_nodes is None or dependent_value in exclusive_nodes):
                        dependent_value_node = self.value_nodes_by_id[dependent_value]
                        child_paths.append(self.get_terminal_source_paths(dependent_value_node, case_index, seen_nodes))
                if len(child_paths) == len(source_node.get_dependent_values()) and all(len(l) > 0 for l in child_paths):
                    for children_combo in product(*child_paths):
                        # Conditions not allowed:
                        # 1. Constructor where all inputs are constants
                        # 2. Attribute where the input is a constructor which has the attribute as an arg - TODO impl is naive
                        if not (source_node.get_source_type() == SourceType.CONSTRUCTOR and all(child_path.source_type == SourceType.CONSTANT for child_path in children_combo)) and \
                            not (source_node.get_source_type() == SourceType.ATTRIBUTE and all(child_path.source_type == SourceType.CONSTRUCTOR for child_path in children_combo)):
                            source_paths.append(PathNode(value_node.id, source_node.id, source_node.get_source_type(), children_combo))

        if exclusive_nodes is not None:
            new_omitted_nodes = omitted_nodes.difference(exclusive_nodes)
        else:
            new_omitted_nodes = set(seen_nodes)

        if cached_paths is not None:
            cached_paths.extend(source_paths)
            value_node.temp_loaded_paths_by_case[case_index] = cached_paths, new_omitted_nodes
        else:
            value_node.temp_loaded_paths_by_case[case_index] = source_paths, new_omitted_nodes

        return source_paths if cached_paths is None else cached_paths


    def terminal_source_path_to_source_graph(self, path_node: PathNode, graph: PortGraph,
                                             input_nodes_by_id: Dict[int, Node],
                                             target: Node, target_port: int = 0,
                                             source_port: Optional[str] = None) -> PortGraph:

        source_node = self.source_nodes_by_id[path_node.source_id]

        if path_node.source_type == SourceType.CONSTANT:
            value_node = self.value_nodes_by_id[path_node.value_id]
            constant_node = Constant(value_node.value_type, copy(value_node.value))
            graph.add_edge(constant_node, target, from_port=source_port, to_port=target_port)
        elif path_node.source_type == SourceType.INPUT:
            if source_node.perceived_value_id is None:
                raise Exception()
            input_node = input_nodes_by_id[source_node.perceived_value_id]
            graph.add_edge(input_node, target, from_port=source_port, to_port=target_port)
        elif path_node.source_type == SourceType.ATTRIBUTE:
            # TODO strongly type the pathnode somehow
            if len(path_node.children) != 1:
                raise Exception()
            concatenated_attr_path = source_node.attribute_path + ("." + source_port if source_port is not None else "")
            self.terminal_source_path_to_source_graph(path_node.children[0], graph, input_nodes_by_id, target,
                                                      target_port=target_port, source_port=concatenated_attr_path)
        elif path_node.source_type == SourceType.CONSTRUCTOR:
            # Create a constant of the type of the constructor
            if source_node.constructor is None:
                raise Exception()
            type_constant = Constant(Type, source_node.constructor)
            constructor_operator = ConstructorOperator()
            graph.add_edge(type_constant, constructor_operator, to_port=0)

            for i, child_path_node in enumerate(path_node.children):
                self.terminal_source_path_to_source_graph(child_path_node, graph, input_nodes_by_id,
                                                          constructor_operator, target_port = i + 1,
                                                          source_port = None)

            graph.add_edge(constructor_operator, target, from_port=source_port, to_port=target_port)
        else:
            raise Exception("Not implemented")

        return graph

    def reset_path_caches(self):
        for value_node in self.value_nodes_by_id.values():
            value_node.temp_loaded_paths_by_case.clear()

    # Get or create the value node
    # Creates attribute sources for children and constructor sources for it.
    def __lazy_create_value_node(self, value_type: Type, value: Any) -> ValueNode:
        search_value_node = ValueNode(value_type, value)
        value_node_id = self.value_nodes_to_id.get(search_value_node)
        if value_node_id is None:
            self.value_nodes_to_id[search_value_node] = search_value_node.id
            self.value_nodes_by_id[search_value_node.id] = search_value_node
            value_node = search_value_node

            # NOT FIGURED OUT YET
            # # All values in the network get a constant source node by default,
            # # but its usage needs to be proven with constraints
            # self.__register_source(TerminalSourceNode(value_node.id, value_type, None, True, None))

            # Create child values, register attribute sources for them, and a constructor source for this value
            # TODO no longer doing constructor - is this needed in the future?
            # self.__add_constructor_value(value_node)
            # for attr_path, attr_type in self.class_component_catalog.get(value_type, {}).items():
            #     attr_value = get_source_port_value(value, attr_path)
            #     self.__add_attribute_value(value_node, attr_type, attr_value, attr_path)
        else:
            value_node = self.value_nodes_by_id[value_node_id]

        return value_node

    def __add_attribute_value(self, parent_value_node: ValueNode, value_type: Type, value: Any,
                              attribute_path: str) -> ValueNode:
        value_node = self.__lazy_create_value_node(value_type, value)
        attribute_source = AttributeSourceNode(value_node.id, value_type, attribute_path, parent_value_node.id)
        self.__register_source(attribute_source, None)
        self.attribute_reverse_source_nodes_by_value_id[parent_value_node.id].add(attribute_source)
        return value_node

    def __register_source(self, source_node: SourceNode, case_index: Optional[int]) -> None:

        existing_source_node_id = self.source_nodes_to_id.get(source_node)

        if existing_source_node_id is None:
            self.source_nodes_to_id[source_node] = source_node.id
            self.source_nodes_by_id[source_node.id] = source_node
            existing_source_node = source_node
        else:
            existing_source_node = self.source_nodes_by_id[existing_source_node_id]

        if case_index is not None:
            self.source_nodes_by_value_id_and_case[(existing_source_node.value_id, case_index)].add(existing_source_node)
        else:
            self.source_nodes_by_value_id_case_agnostic[existing_source_node.value_id].add(existing_source_node)
