from collections import defaultdict
from typing import List, DefaultDict, Set, Dict, Iterable

from nodes import RelationshipNode, ProxyNode, SetJoin, DisjointSetNode, Node
from port_graphs import Edge, PortGraph


class OperatorPlan:

    def __init__(self, root_nodes: List[int], first_relationships: List[int],
                 non_constraint_nodes: List[int], join_nodes: Iterable[int],):
        self.__root_nodes = root_nodes
        self.__first_relationships = first_relationships
        # Shortest path to each set join first.
        self.__non_constraint_nodes = non_constraint_nodes
        self.__join_nodes = join_nodes

    def get_root_nodes(self) -> List[int]:
        return self.__root_nodes

    def get_immediate_relationships(self) -> List[int]:
        return self.__first_relationships

    def get_non_constraint_nodes(self) -> List[int]:
        return self.__non_constraint_nodes

    def get_join_nodes(self) -> List[int]:
        return self.__join_nodes

    @classmethod
    def get_operator_plan(cls, graph: PortGraph) -> 'OperatorPlan':

        root_node_ids: List[int] = graph.get_root_node_ids()
        all_node_layers: DefaultDict[int, Set[int]] = defaultdict(set)

        # Traverse the graph bfs, keep track of what "layer" currently at, store all layers seen by each node.
        queue: List[List[int]] = [root_node_ids]
        seen_edges: Set[Edge] = set()

        layer = 0
        while queue:
            next_layer = []
            layer_nodes = queue.pop(0)
            for node_id in layer_nodes:
                all_node_layers[node_id].add(layer)
                for edge in graph.get_edges_from_by_id(node_id):
                    if edge not in seen_edges or not (isinstance(graph.get_node_by_id(edge.target_node), ProxyNode)
                                                      and edge.target_port == 1):
                        next_layer.append(edge.target_node)
                        seen_edges.add(edge)

            if next_layer:
                queue.append(next_layer)
            layer += 1

        max_node_layers: Dict[int, int] = {node_id: max(layers) for node_id, layers in all_node_layers.items()}
        layers_to_nodes = [[] for _ in range(layer)]
        for node_id, max_layer in max_node_layers.items():
            layers_to_nodes[max_layer].append(node_id)

        non_constraint_nodes: List[Node] = []
        immediate_relationships: List[Node] = []
        for layer_1_node in layers_to_nodes[1]:
            if isinstance(graph.get_node_by_id(layer_1_node), RelationshipNode):
                immediate_relationships.append(layer_1_node)
            else:
                non_constraint_nodes.append(layer_1_node)

        if len(layers_to_nodes) > 2:
            for layer_nodes in layers_to_nodes[2:]:
                non_constraint_nodes.extend(sorted(layer_nodes))

        # Duplicate all recursive nodes, adding them to the end so they are evaluated a second time
        for node in graph.get_nodes_by_id().values():
            if isinstance(node, ProxyNode):
                non_constraint_nodes.append(node.id)

        join_nodes = ([node.id for node in graph.get_nodes_by_type(SetJoin.__name__)] +
                      [node.id for node in graph.get_nodes_by_type(DisjointSetNode.__name__)
                       if node.id not in root_node_ids])

        return cls(root_node_ids, immediate_relationships, non_constraint_nodes, join_nodes)
