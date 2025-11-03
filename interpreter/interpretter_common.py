from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, DefaultDict, Iterable, Set

from base_types import DslSet, get_source_port_value
from nodes import OutputNode, DisjointSetNode, Node
from port_graphs import PortGraph


class GraphInstanceState:

    __global_id = 0

    @staticmethod
    def get_next_id():
        GraphInstanceState.__global_id += 1
        return GraphInstanceState.__global_id

    def __init__(self, graph: PortGraph, state: Dict[int, Any], generic_index: Optional[int]):
        self.graph = graph
        self.state = state
        self.generic_index = generic_index
        self.last_evaluated_node_id = None
        self.complete = False

        self.id = GraphInstanceState.get_next_id()
        self.parent_id = None

    def __copy__(self):
        new_state = {}
        for key, value in self.state.items():
            if isinstance(value, DslSet):
                iterable_type = type(value)
                new_state[key] = iterable_type(e for e in value)
            else:
                new_state[key] = value  # Copy other values directly
        return GraphInstanceState(self.graph, new_state, self.generic_index)

    def get_state_hash(self):
        return hash((self.generic_index, tuple((k, self.__hash_cast_state_value_safe(v))
                                               for k, v in sorted(self.state.items(), key=lambda kv: kv[0]))))

    @staticmethod
    def __hash_cast_state_value_safe(value: Any):
        if isinstance(value, DslSet):
            return tuple(sorted(value, key=lambda v: hash(GraphInstanceState.__hash_cast_state_value_safe(v))))
        return value


class GraphIO:

    def __init__(self, graph: PortGraph):
        self._graph = graph
        self._input_values_by_node_id: DefaultDict[int, List[Any]] = defaultdict(list)
        self._output_values_by_node_id: DefaultDict[int, List[Any]] = defaultdict(list)
        self._disjoint_sets_by_node_id: Dict[int, DslSet] = defaultdict(DslSet)

    def add_input_value(self, node_id:int, object_instance: Any):
        self._input_values_by_node_id[node_id].append(object_instance)

    def extend_input_values(self, node_id:int, object_instances: Iterable[Any]):
        self._input_values_by_node_id[node_id].extend(object_instances)

    def get_input_values(self, node_id:int):
        return self._input_values_by_node_id[node_id]

    def collect_output_values(self, graph_instances: Iterable[GraphInstanceState]):
        # Must check nodes with isinstance to ensure we get subclasses.
        output_node_ids = []
        for node in self._graph.get_nodes_by_id().values():
            if isinstance(node, OutputNode):
                output_node_ids.append(node.id)

        for output_node_id in output_node_ids:
            for graph_instance in graph_instances:
                value = graph_instance.state.get(output_node_id)
                if value:
                    self._output_values_by_node_id[output_node_id].append(value)

    def get_output_values(self) -> Dict[int, List[Any]]:
        return self._output_values_by_node_id

    def set_disjoint_set(self, node_id: int, collected_set: DslSet):
        self._disjoint_sets_by_node_id[node_id] = collected_set

    def get_disjoint_set(self, node_id: int):
        return self._disjoint_sets_by_node_id[node_id]

def get_sorted_edges_to(graph: PortGraph, node_id: int, generic_index: Optional[int]):
    return sorted(graph.get_edges_to_by_id(node_id, generic_index), key=lambda e: e.target_port)

def get_node_port_tuples(subgraph: PortGraph, node_id: int, generic_index: Optional[int]):
    return [(edge.source_node, edge.source_port) for edge in get_sorted_edges_to(subgraph, node_id, generic_index)]


def get_args(subgraph_dict: Dict[int, Any], node_port_tuples: List[Tuple[int, str]]):
    return [get_source_port_value(subgraph_dict.get(index), port) for index, port in node_port_tuples]


def get_args_immediate(subgraph: PortGraph, graph_instance: GraphInstanceState, node_id: int):
    subgraph_dict: Dict[int, Any] = graph_instance.state
    node_port_tuples = get_node_port_tuples(subgraph, node_id, graph_instance.generic_index)
    return get_args(subgraph_dict, node_port_tuples)


def traverse_connected_graph(graph: PortGraph, entry_point: Node, visited: Set[int]):

    if entry_point.id in visited:
        return

    visited.add(entry_point.id)

    # Disjoint set nodes do not contribute to the "connectedness" of the subgraph.
    if not isinstance(entry_point, DisjointSetNode):
        for edge in graph.get_edges_to_by_id(entry_point.id):
            traverse_connected_graph(graph, graph.get_node_by_id(edge.source_node), visited)

        for edge in graph.get_edges_from_by_id(entry_point.id):
            traverse_connected_graph(graph, graph.get_node_by_id(edge.target_node), visited)
