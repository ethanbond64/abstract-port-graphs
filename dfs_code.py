from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Tuple, List, Set, Callable, Dict, Union

from nodes import Constant, RelationshipNode, OperatorNode, Node, InputNode, OutputNode
from port_graphs import Edge, PortGraph

GENERIC_EDGE_FROM_LABEL = "0"
GENERIC_EDGE_TO_LABEL = "1"

# SET OF NODE TYPES WHICH HAVE A GENERIC COUNTERPART
VALID_GENERIC_NODE_TYPES: Set[str] = {
    Constant,
    RelationshipNode,
    OperatorNode
}

class DfsCodeSchema(Enum):
    SPECIFIC = 0
    GENERIC = 1


def specific_node_label_function(node: Node) -> str:
    if isinstance(node, Constant):
        return Constant.__name__ + "(" + str(node.value_type) + ", " + str(hash(node.value)) + ")"
    return node.__class__.__name__


def specific_edge_label_function(edge: Edge, edge_from: bool) -> str:
    if edge_from:
        return str((edge.source_port, edge.target_port))
    else:
        return str((edge.target_port, edge.source_port))


def generic_node_label_function(node: Node) -> str:
    if isinstance(node, Constant):
        return Constant.__name__ + "(" + str(node.value_type) + ")"
    return node.__class__.__name__


def generic_edge_label_function(edge: Edge, edge_from: bool):
    return GENERIC_EDGE_FROM_LABEL if edge_from else GENERIC_EDGE_TO_LABEL


# MAP OF DFS CODE SCHEMA TO TUPLE (NODE LABEL FUNCTION, EDGE LABEL FUNCTION)
DFS_SCHEMA_MAP = {
    DfsCodeSchema.SPECIFIC: (specific_node_label_function, specific_edge_label_function),
    DfsCodeSchema.GENERIC: (generic_node_label_function, generic_edge_label_function)
}


def get_other_node(graph: PortGraph, edge: Edge, edge_from: bool) -> Node:
    if edge_from:
        other_node_id = edge.target_node
    else:
        other_node_id = edge.source_node

    return graph.get_node_by_id(other_node_id)


class DfsCode:

    def __init__(self, root_node: Node, node_label_fn: Callable, edge_label_fn: Callable):

        self.__root_node = root_node
        # TODO cache results from these fns
        self.__node_label_fn = node_label_fn
        self.__edge_label_fn = edge_label_fn

        # Tuple of (int, int, str, str, str)
        # Semantically (from_dfs_id, to_dfs_id, from_label, edge_label, to_label)
        self.__tuples: List[Tuple] = []

        # List of (external) node ids. The internal dfs node id comes from the index in this list.
        # AKA Node Discovery Time...
        self.__node_dfs_ids: List[int] = [root_node.id]
        self.__node_dfs_stack: List[int] = [root_node.id]
        self.__edge_set: Set[Edge] = set()

        # Used when creating a generic graph between two common dfs codes + graphs
        self.__tuple_to_edges: Dict[Tuple, Edge] = {}

        # toString caching
        self.__str = ""
        self.__str_tuple_length = 0

    def add_edge_tuple(self, from_node: Node, edge: Edge, edge_from: bool, to_node: Node):

        if edge in self.__edge_set:
            raise Exception()

        source_node_dfs_id = self.__node_dfs_ids.index(from_node.id)
        target_node_dfs_id = self.__lazy_get_dfs_id(to_node.id)
        source_label = self.__node_label_fn(from_node)
        edge_label = self.__edge_label_fn(edge, edge_from)
        target_label = self.__node_label_fn(to_node)

        edge_tuple = (source_node_dfs_id, target_node_dfs_id, source_label, edge_label, target_label)
        self.__tuples.append(edge_tuple)
        self.__edge_set.add(edge)
        self.__tuple_to_edges[edge_tuple] = edge

    def get_current_node(self):
        return self.__node_dfs_stack[-1]

    def seen_edge(self, edge: Edge):
        return edge in self.__edge_set

    def split_edges_backwards_forwards(self, edge_list: List[Tuple[Edge, bool]]):
        backwards = []
        forwards = []

        for edge_tuple in edge_list:
            edge, from_edge = edge_tuple
            other_node_id = edge.target_node if from_edge else edge.source_node

            if other_node_id in self.__node_dfs_ids:
                other_node_dfs_id = self.__node_dfs_ids.index(other_node_id)
                backwards.append((edge_tuple, other_node_dfs_id))
            else:
                forwards.append(edge_tuple)

        # Sort backwards by canonical index of the other node
        backwards = list(map(lambda t: t[0], sorted(backwards, key=lambda t: t[1])))

        return backwards, forwards

    def backtrack(self):
        self.__node_dfs_stack.pop()

    def get_depth(self):
        return len(self.__tuples)

    def is_complete(self):
        return len(self.__node_dfs_stack) == 0

    def get_string(self) -> str:

        if len(self.__tuples) == self.__str_tuple_length:
            return self.__str

        for t in self.__tuples[self.__str_tuple_length:]:
            self.__str += str(t)

        # self.__str = "".join(str(t) for t in self.__tuples)

        self.__str_tuple_length = len(self.__tuples)

        return self.__str

    def get_node_dfs_id(self, external_node_id: int):
        return self.__node_dfs_ids.index(external_node_id)

    def get_node_real_id(self, dfs_id: int):
        return self.__node_dfs_ids[dfs_id]

    def get_edge_tuples(self) -> List[Tuple]:
        return self.__tuples

    def get_real_edge(self, tup: Tuple) -> Edge:
        return self.__tuple_to_edges[tup]

    def __lazy_get_dfs_id(self, external_node_id):
        try:
            return self.__node_dfs_ids.index(external_node_id)
        except ValueError:
            self.__node_dfs_ids.append(external_node_id)
            self.__node_dfs_stack.append(external_node_id)
            return len(self.__node_dfs_ids) - 1

    def __deepcopy__(self):
        new_instance = DfsCode(self.__root_node, self.__node_label_fn, self.__edge_label_fn)
        # Tuple of (int, int, str, str, str)
        # Semantically (from_dfs_id, to_dfs_id, from_label, edge_label, to_label)
        new_instance.__tuples = list(self.__tuples)
        new_instance.__node_dfs_ids = list(self.__node_dfs_ids)
        new_instance.__node_dfs_stack = list(self.__node_dfs_stack)
        new_instance.__edge_set = set(self.__edge_set)
        new_instance.__tuple_to_edges = dict(self.__tuple_to_edges)

        new_instance.__str = self.__str
        new_instance.__str_tuple_length = self.__str_tuple_length
        return new_instance

# Start with an empty dfs code
# Search the graph for the min label nodes
# Create a child dfs code for each node with the same label
# Enqueue all child nodes for bfs
# Track min dfs code str so far
# Track the bfs depth
# LOOP
# Pop a dfs code off the queue
# Compare parent to min dfs code if depth is diff OR find best fit children, and queue them up iff be


def get_min_dfs_code(graph: PortGraph, schema: DfsCodeSchema) -> DfsCode:

    node_label_fn, edge_label_fn = DFS_SCHEMA_MAP[schema]

    queue: List[DfsCode] = []
    min_code = None

    # Create roots for equivalent starting points.
    label_to_nodes = defaultdict(list)
    for node in graph.get_nodes_by_id().values():
        label = node_label_fn(node)
        label_to_nodes[label].append(node)
    min_label = min(label_to_nodes.keys())
    starter_nodes: List[Node] = label_to_nodes[min_label]
    for starter_node in starter_nodes:
        queue.append(DfsCode(starter_node, node_label_fn, edge_label_fn))

    while queue:

        current_dfs_code = queue.pop(0)

        if min_code:
            if min_code.get_depth() == current_dfs_code.get_depth():
                if current_dfs_code.get_string() < min_code.get_string():
                    min_code = current_dfs_code
                elif current_dfs_code.get_string() > min_code.get_string():
                    continue
            elif min_code.get_depth() == current_dfs_code.get_depth() - 1:
                min_code = current_dfs_code
            else:
                # TODO confirm...
                continue
                # raise Exception()
        else:
            min_code = current_dfs_code

        current_node_id, all_edges = get_current_node_and_edges(current_dfs_code, graph)

        while not all_edges and not current_dfs_code.is_complete():
            current_dfs_code.backtrack()
            if not current_dfs_code.is_complete():
                current_node_id, all_edges = get_current_node_and_edges(current_dfs_code, graph)

        if current_dfs_code.is_complete():
            # TODO duplicate code below
            if min_code.get_depth() == current_dfs_code.get_depth():
                if current_dfs_code.get_string() < min_code.get_string():
                    min_code = current_dfs_code
                elif current_dfs_code.get_string() > min_code.get_string():
                    continue
            elif min_code.get_depth() == current_dfs_code.get_depth() - 1:
                min_code = current_dfs_code
            else:
                raise Exception()
            # TODO duplicate code above

        # Split into forward edges and backward edges
        backward_edges, forward_edges = current_dfs_code.split_edges_backwards_forwards(all_edges)

        if backward_edges:
            edge, edge_from = backward_edges[0]
            current_node = graph.get_node_by_id(current_node_id)
            target_node = get_other_node(graph, edge, edge_from)
            current_dfs_code.add_edge_tuple(current_node, edge, edge_from, target_node)
            queue.append(current_dfs_code)
        elif forward_edges:
            # Find the min label... find all edges which provide the same label... enqueue COPIES for each
            label_to_edge_tuple_args = defaultdict(list)
            current_node = graph.get_node_by_id(current_node_id)
            current_label = node_label_fn(current_node)
            for forward_edge_info in forward_edges:
                edge_label = edge_label_fn(*forward_edge_info)
                target_node = get_other_node(graph, *forward_edge_info)
                target_label = node_label_fn(target_node)
                composite_label = str((current_label, edge_label, target_label))
                label_to_edge_tuple_args[composite_label].append((current_node, *forward_edge_info, target_node))

            min_label = min(label_to_edge_tuple_args.keys())
            for args in label_to_edge_tuple_args[min_label]:
                new_dfs_code = deepcopy(current_dfs_code)
                new_dfs_code.add_edge_tuple(*args)
                queue.append(new_dfs_code)

    return min_code


def get_current_node_and_edges(current_dfs_code, graph):
    current_node_id = current_dfs_code.get_current_node()
    # Get edges touching the current node
    # Tuple of edge and boolean "from side"
    all_edges = []
    all_edges.extend((e, False) for e in graph.get_edges_to_by_id(current_node_id))
    all_edges.extend((e, True) for e in graph.get_edges_from_by_id(current_node_id))
    # Filter out edges already tracked
    all_edges = list(filter(lambda t: not current_dfs_code.seen_edge(t[0]), all_edges))
    return current_node_id, all_edges


# Each mapping corresponds to an arg tuple, with a dict of the corresponding node from the original arg
def merge_graphs_to_generic_graph(graph_dfs_tuples: List[Tuple[PortGraph, DfsCode]]) -> Tuple[PortGraph, List[Dict[int, int]]]:

    if any(t[0].has_generics() for t in graph_dfs_tuples):
        raise Exception("Cannot merge generic graph into greater generic graph yet.")

    generic_graph = PortGraph()
    generic_node_cache: Dict[int, Union[Node, List[Node]]] = {}

    def get_generic_node(dfs_id) -> Union[Node, List[Node]]:

        existing = generic_node_cache.get(dfs_id)
        if existing is not None:
            return existing

        # Iterate all dfs codes, compare classes, special handling for constants (compare values)
        # Dedup node ids by ALWAYS refreshing
        concrete_node_list = [graph.get_node_by_id(dfs.get_node_real_id(dfs_id)).copy_and_refresh_id()[0]
                              for graph, dfs in graph_dfs_tuples]
        main_node = concrete_node_list[0]
        if all(main_node.__class__ == node.__class__ and
               (not isinstance(main_node, Constant) or main_node.value == node.value) for node in concrete_node_list):
            generic_node_cache[dfs_id] = main_node
            return main_node

        generic_node_cache[dfs_id] = concrete_node_list
        return concrete_node_list

    driver_dfs: DfsCode = graph_dfs_tuples[0][1]
    dfs_id_to_final_node_id: Dict[int, int] = {}

    for edge_tuple in driver_dfs.get_edge_tuples():
        first_dfs_id = edge_tuple[0]
        first_node = get_generic_node(edge_tuple[0])

        second_dfs_id = edge_tuple[1]
        second_node = get_generic_node(edge_tuple[1])

        source_ports = []
        target_ports = []
        for dfs_code in map(lambda tup: tup[1], graph_dfs_tuples):
            real_edge = dfs_code.get_real_edge(edge_tuple)
            source_ports.append(real_edge.source_port)
            target_ports.append(real_edge.target_port)

        source_port = source_ports[0] if all(source_ports[0] == sp for sp in source_ports) else source_ports
        target_port = target_ports[0] if all(target_ports[0] == tp for tp in target_ports) else target_ports

        if edge_tuple[3] == GENERIC_EDGE_FROM_LABEL:
            from_node = first_node
            from_dfs_id = first_dfs_id
            to_node = second_node
            to_dfs_id = second_dfs_id
        elif edge_tuple[3] == GENERIC_EDGE_TO_LABEL:
            from_node = second_node
            from_dfs_id = second_dfs_id
            to_node = first_node
            to_dfs_id = first_dfs_id
        else:
            raise Exception("Invalid edge label for generics")

        generic = any(isinstance(o, list) for o in (from_node, to_node, source_port, target_port))
        if generic:
            new_edge = generic_graph.add_generic_edge(from_node, to_node, source_port, target_port)
        else:
            new_edge = generic_graph.add_edge(from_node, to_node, source_port, target_port)

        # Track the created dsl nodes by the dfs id to create mappings in the end
        if from_dfs_id in dfs_id_to_final_node_id:
            if dfs_id_to_final_node_id[from_dfs_id] != new_edge.source_node:
                raise Exception("Inconsistent generic merge")
        else:
            dfs_id_to_final_node_id[from_dfs_id] = new_edge.source_node

        if to_dfs_id in dfs_id_to_final_node_id:
            if dfs_id_to_final_node_id[to_dfs_id] != new_edge.target_node:
                raise Exception("Inconsistent generic merge")
        else:
            dfs_id_to_final_node_id[to_dfs_id] = new_edge.target_node

    node_lookups = [{ dfs_tuple[1].get_node_real_id(dfs_id): final_node_id
                      for dfs_id, final_node_id in dfs_id_to_final_node_id.items() }
                    for dfs_tuple in graph_dfs_tuples]

    return generic_graph, node_lookups

# if __name__ == "__main__":
#     start = time.time()
    # g = solve_09629e4f()
    # code = get_min_dfs_code(g)
    # print(code.get_depth())
    # end = time.time()
    # print(code.get_string())
    # print(end-start)

