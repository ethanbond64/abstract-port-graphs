from collections import defaultdict
from copy import deepcopy, copy
from typing import List, Union, Any, Dict, Set, DefaultDict, Tuple, Optional

from nodes import Node, Constant, GenericConstantNode, RelationshipNode, GenericRelationshipNode, \
    OperatorNode, GenericOperatorNode, GenericBaseNode


class Edge:

    def __init__(self, source_node, target_node, source_port, target_port):
        self.source_node: int = source_node
        self.target_node: int = target_node
        self.source_port: str = source_port
        self.target_port: int = target_port

    def get_edge_label(self):
        return str((self.source_port, self.target_port))

    def get_debug_label(self):
        src = "" if self.source_port is None else str(self.source_port)
        tgt = "" if self.target_port is None else str(self.target_port)

        if src == "" and tgt == "0":
            return ""

        return f"{src} -> {tgt}"

    def __eq__(self, __value):
        return (self.source_node == __value.source_node and
                self.target_node == __value.target_node and
                self.source_port == __value.source_port and
                self.target_port == __value.target_port)

    def __hash__(self):
        return hash((self.source_node, self.target_node, self.source_port, self.target_port))


class GenericEdge(Edge):

    def __init__(self, source_node: int, target_node: int, generic_source_port: Union[str,List[str]] = None,
                 generic_target_port: Union[int,List[int]] = 0):
        super().__init__(source_node, target_node, None, None)
        self.generic_source_port = generic_source_port
        self.hashable_source_port = tuple(generic_source_port) if isinstance(generic_source_port, list) else generic_source_port
        self.generic_target_port = generic_target_port
        self.hashable_target_port = tuple(generic_target_port) if isinstance(generic_target_port, list) else generic_target_port

    def concrete(self, generic_index):
        source_port = self.generic_source_port[generic_index] if isinstance(self.generic_source_port, list) else self.generic_source_port
        target_port = self.generic_target_port[generic_index] if isinstance(self.generic_target_port, list) else self.generic_target_port
        return Edge(self.source_node, self.target_node, source_port, target_port)

    def __eq__(self, __value):
        return (self.source_node == __value.source_node and
                self.target_node == __value.target_node and
                self.hashable_source_port == __value.hashable_source_port and
                self.hashable_target_port == __value.hashable_target_port)

    def __hash__(self):
        return hash((self.source_node, self.target_node, self.hashable_source_port, self.hashable_target_port))


class PortGraph:

    MAINTAIN_CONSTANT = "__maintain__"

    def __init__(self, initial_nodes: List[Node] = None):
        self.__nodes_by_id: Dict[int, Node] = {}
        self.__nodes_by_type: Dict[str, Set[Node]] = defaultdict(set)
        self.__edge_set: Set[Edge] = set()
        self.__edges_to_by_id: DefaultDict[int, Set[Edge]] = defaultdict(set)
        self.__edges_from_by_id: DefaultDict[int, Set[Edge]] = defaultdict(set)

        # Lookup from concrete node ids (in order) to generic placeholder id
        self.__generic_node_reverse_lookup: Dict[Tuple[int, ...], int] = {}
        # Lookup from concrete node id (single) to generic placeholder id
        self.__generic_node_single_lookup: Dict[int, int] = {}
        # Lookup from generic placeholder id to concrete node ids (in order)
        self.__generic_node_lookup: Dict[int, Tuple[int, ...]] = {}
        self.__generic_nodes_by_id: Dict[int, Node] = {}

        self.__generic_case_count: int = 0

        if initial_nodes:
            for initial_node in initial_nodes:
                self.__nodes_by_id[initial_node.id] = initial_node
                self.__nodes_by_type[initial_node.__class__.__name__].add(initial_node)

    def has_generics(self):
        return self.__generic_case_count > 0

    def get_generic_count(self):
        return self.__generic_case_count

    def adopt_generics(self, child_graph):
        self.__generic_node_reverse_lookup = child_graph.__generic_node_reverse_lookup
        self.__generic_node_single_lookup = child_graph.__generic_node_single_lookup
        self.__generic_node_lookup = child_graph.__generic_node_lookup
        self.__generic_nodes_by_id = child_graph.__generic_nodes_by_id
        self.__generic_case_count = child_graph.__generic_case_count

    def add_node(self, node: Node):
        self.__nodes_by_id[node.id] = node
        self.__nodes_by_type[node.__class__.__name__].add(node)

    def add_edge(self, from_node: Node, to_node: Node, from_port: str = None, to_port: int = 0) -> Edge:
        from_node = self._add_node(from_node)
        to_node = self._add_node(to_node)
        edge = Edge(from_node.id, to_node.id, from_port, to_port)
        self.add_existing_edge(from_node, to_node, edge)

        return edge

    def add_generic_edge(self, from_node: Union[Node, List[Node]], to_node: Union[Node, List[Node]],
                         from_port: Union[str, List[str]] = None,
                         to_port: Union[int, list[int]] = 0) -> Union[GenericEdge, Edge]:
        # TODO remove this validation for performance in the future - this whole method could be optimized
        for arg in (from_node, to_node, from_port, to_port):
            if isinstance(arg, List):
                self.__validate_generic_list(arg)

        if self.__generic_case_count is None:
            print("add_generic_edge method reserved for generic cases only")
            raise Exception()

        generic_from_node = from_node if isinstance(from_node,
                                                    Node) else self.__validate_and_lazy_create_generic_node(from_node)
        generic_to_node = to_node if isinstance(to_node,
                                                Node) else self.__validate_and_lazy_create_generic_node(to_node)

        if isinstance(from_port, List) or isinstance(to_port, List):
            generic_edge = GenericEdge(generic_from_node.id, generic_to_node.id, from_port, to_port)
        else:
            generic_edge = Edge(generic_from_node.id, generic_to_node.id, from_port, to_port)
        self.add_existing_edge(generic_from_node, generic_to_node, generic_edge)

        return generic_edge

    # Validates that the list is a valid size (throws error if not).
    # OR lazy sets the generic count if first list encountered
    def __validate_generic_list(self, arg_list: List[Any]):
        if self.__generic_case_count == 0:
            if len(arg_list) < 2:
                print("Generic list too short")
                raise Exception()
            self.__generic_case_count = len(arg_list)
        elif len(arg_list) != self.__generic_case_count:
            print("Invalid generic list length")
            raise Exception("Invalid generic list length")

    def __validate_and_lazy_create_generic_node(self, concrete_node_list: List[Node]):
        if any(node.id in self.__nodes_by_id.keys() for node in concrete_node_list):
            print("Generic's concrete node cannot be in the graph")
            raise Exception("Generic concrete node cannot be in the graph")

        key: Tuple[int, ...] = tuple(node.id for node in concrete_node_list)
        if any(node.id in self.__generic_nodes_by_id.keys() for node in concrete_node_list):
            if not all(node.id in self.__generic_nodes_by_id.keys() for node in concrete_node_list):
                print("Generic concrete node cannot be reused in a different generic node")
                raise Exception("Generic concrete node be cannot reused in a different generic node")
            return self.__nodes_by_id[self.__generic_node_reverse_lookup[key]]

        # Create the new generic node
        if all(isinstance(node, Constant) for node in concrete_node_list):
            generic_node = GenericConstantNode()
        elif all(isinstance(node, RelationshipNode) for node in concrete_node_list):
            generic_node = GenericRelationshipNode()
        elif all(isinstance(node, OperatorNode) for node in concrete_node_list):
            generic_node = GenericOperatorNode()
        else:
            print("Unsupported node type for generic concrete node ", concrete_node_list,
                  [type(node) for node in concrete_node_list])
            raise Exception("Unsupported node type for generic concrete node")

        self.__generic_node_reverse_lookup[key] = generic_node.id
        self.__generic_node_lookup[generic_node.id] = key
        for node in concrete_node_list:
            self.__generic_nodes_by_id[node.id] = node
            self.__generic_node_single_lookup[node.id] = generic_node.id

        return generic_node

    def add_existing_edge(self, from_node: Node, to_node: Node, edge: Edge):
        from_node = self._add_node(from_node)
        to_node = self._add_node(to_node)
        self.__edge_set.add(edge)
        self.__edges_to_by_id[to_node.id].add(edge)
        self.__edges_from_by_id[from_node.id].add(edge)

    def get_nodes_by_id(self) -> Dict[int, Node]:
        return self.__nodes_by_id

    def get_node_by_id(self, node_id: int, generic_index=None) -> Node:
        # case normal node - return it
        # case generic node and node index integer - get concrete id and return it
        # case generic node and null index - return concrete node
        # case concrete node... return concrete node

        if node_id in self.__generic_nodes_by_id.keys():
            return self.__generic_nodes_by_id[node_id]
        elif generic_index is None or node_id not in self.__generic_node_lookup.keys():
            return self.__nodes_by_id[node_id]

        concrete_node_ids = self.__generic_node_lookup[node_id]
        return self.__generic_nodes_by_id[concrete_node_ids[generic_index]]

    def get_nodes_by_type(self, node_type: str) -> Set[Node]:
        return self.__nodes_by_type[node_type]

    def get_edge_set(self) -> Set[Edge]:
        return self.__edge_set

    def get_edges_to_by_id(self, node_id: int, generic_index=None) -> Set[Edge]:
        return self.__get_edge_by_id_common(self.__edges_to_by_id, node_id, generic_index)

    def get_edges_from_by_id(self, node_id: int, generic_index=None) -> Set[Edge]:
        return self.__get_edge_by_id_common(self.__edges_from_by_id, node_id, generic_index)

    def get_root_node_ids(self) -> List[int]:
        return [node_id for node_id in self.__nodes_by_id.keys() if node_id not in self.__edges_to_by_id]

    def get_all_generic_nodes(self) -> List[Node]:
        return self.__generic_nodes_by_id.values()

    def is_node_generic(self, node_id: int) -> bool:
        if node_id not in self.__nodes_by_id.keys():
            raise Exception("Node id not in graph")
        return node_id in self.__generic_node_lookup.keys() or isinstance(self.__nodes_by_id[node_id], GenericBaseNode)

    def _add_node(self, node: Node) -> Node:
        self.__nodes_by_id[node.id] = node
        self.__nodes_by_type[node.__class__.__name__].add(node)
        return node

    def __get_edge_by_id_common(self, edge_collection: DefaultDict[int, Set[Edge]], node_id: int, generic_index=None):

        if node_id in self.__nodes_by_id.keys():
            pre_cleaned_edges = edge_collection[node_id]
        else:
            pre_cleaned_edges = edge_collection[self.__generic_node_single_lookup[node_id]]

        return set(edge.concrete(generic_index) if generic_index is not None and isinstance(edge, GenericEdge) else edge for edge in pre_cleaned_edges)

    # NOTE VERY DANGEROUS - DO NOT USE WITH INTERPRETER - DESIGN TIME ONLY!
    def replace_node(self, original_node: Node, new_node: Node, source_port: Optional[str] = None,
                     target_port: Optional[int] = None, maintain_multiple_edges = False, edge_to = False):

        original_node_copy = self.get_node_by_id(original_node.id)
        if original_node_copy is None:
            raise Exception("Original node does not belong to this graph")

        if self.has_generics():
            raise Exception("Generics not supported for replacement")

        new_node.id = original_node.id

        self.__nodes_by_id[new_node.id] = new_node
        self.__nodes_by_type[original_node.__class__.__name__].remove(original_node_copy)
        self.__nodes_by_type[new_node.__class__.__name__].add(new_node)

        if not maintain_multiple_edges:

            edge_list = self.__edges_to_by_id if edge_to else self.__edges_from_by_id

            if len(edge_list[new_node.id]) > 1:
                raise Exception("Multiple outbound edges not supported for replacement")

            edge = list(edge_list[new_node.id])[0]
            if source_port != PortGraph.MAINTAIN_CONSTANT:
                edge.source_port = source_port
            if target_port is not None:
                edge.target_port = target_port

    # TODO lots of duplication between above and below methods. Come back and cleanup at some point
    def replace_node_with_existing_node(self, original_node: Node, new_node: Node,
                                        source_port: Optional[str] = None):

        original_node_copy = self.get_node_by_id(original_node.id)
        if original_node_copy is None:
            raise Exception("Original node does not belong to this graph")

        new_node_copy = self.get_node_by_id(new_node.id)
        if new_node_copy is None:
            raise Exception("New node does not belong to this graph")

        if self.has_generics():
            raise Exception("Generics not supported for replacement")

        del self.__nodes_by_id[original_node.id]
        self.__nodes_by_type[original_node.__class__.__name__].remove(original_node_copy)

        # TODO implement like method below
        edge = list(self.__edges_from_by_id[new_node.id])[0]
        edge.source_port = source_port
        edge.source_node = new_node_copy.id

    def replace_node_with_existing_node_all_edges(self, original_node_id: int, new_node_id: int):

        original_node = self.get_node_by_id(original_node_id)
        if original_node is None:
            raise Exception("Original node does not belong to this graph")

        new_node = self.get_node_by_id(new_node_id)
        if new_node is None:
            raise Exception("New node does not belong to this graph")

        if self.has_generics():
            raise Exception("Generics not supported for replacement")

        del self.__nodes_by_id[original_node.id]
        self.__nodes_by_type[original_node.__class__.__name__].remove(original_node)

        if len(self.__edges_to_by_id[original_node.id]) > 0:
            raise Exception("Inbound edges not supported for replacement")

        for edge in self.__edges_from_by_id[original_node.id]:
            edge.source_node = new_node_id
            self.__edges_from_by_id[new_node_id].add(edge)

        del self.__edges_from_by_id[original_node.id]
        del self.__edges_to_by_id[original_node.id]

    def replace_node_with_new_node_and_inbound_edges(self, original_node: Node, new_node: Node):
        original_node_copy = self.get_node_by_id(original_node.id)
        if original_node_copy is None:
            raise Exception("Original node does not belong to this graph")

        if self.has_generics():
            raise Exception("Generics not supported for replacement")

        new_node.id = original_node.id

        self.__nodes_by_id[new_node.id] = new_node
        self.__nodes_by_type[original_node.__class__.__name__].remove(original_node_copy)
        self.__nodes_by_type[new_node.__class__.__name__].add(new_node)


    # Returns a new merged graph and a lookup telling which node ids were replaced from graph 1 and
    # the ids that replaced them.
    @staticmethod
    def merge_graphs(graph_1: 'PortGraph', graph_2: 'PortGraph', common_ids_1_to_2: Dict[int, int]) -> Tuple[
        'PortGraph', Dict[int, int]]:

        # GENERICS OF DIFFERENT SIZE NOT SUPPORTED YET
        if graph_1.has_generics() and graph_2.has_generics() and graph_1.get_generic_count() != graph_2.get_generic_count():
            raise Exception("Generics of different size not supported for merge yet")

        # COMMON IDS KEY MUST BE BIJECTIVE
        graph_1_common_ids = set(common_ids_1_to_2.keys())
        graph_2_common_ids = set(common_ids_1_to_2.values())
        if len(graph_1_common_ids) != len(graph_2_common_ids):
            raise Exception("Common ids from graph 2 must be unique")

        graph_1_ids = set(graph_1.get_nodes_by_id().keys())
        graph_2_ids = set(graph_2.get_nodes_by_id().keys())

        # ALL PRESCRIBED COMMON IDS MUST EXIST IN THEIR RESPECTIVE GRAPHS
        if (any(common_id not in graph_1_ids for common_id in graph_1_common_ids) or
            any(common_id not in graph_2_ids for common_id in graph_2_common_ids)):
            raise Exception("Common ids must exist")

        # COMMON IDS CANNOT BE GENERIC NODES FOR NOW
        if (any(graph_1.is_node_generic(common_id) for common_id in graph_1_common_ids) or
            any(graph_2.is_node_generic(common_id) for common_id in graph_2_common_ids)):
            raise Exception("Common ids cannot be generic nodes")

        new_graph = deepcopy(graph_2)
        claimed_ids = graph_2_ids.union(set(n.id for n in graph_2.get_all_generic_nodes()))

        # Ids that collided illegally and had to be replaced -> the new node that replaces them
        duplicate_id_old_to_new_lookup = {}

        # Case graph 1 (non-copy) has only disjoint nodes and no edges
        if len(graph_1.get_edge_set()) == 0:
            for node in graph_1.get_nodes_by_id().values():
                node_common_id: Optional[int] = common_ids_1_to_2.get(node.id)
                node_dedup_id: Optional[int] = duplicate_id_old_to_new_lookup.get(node.id)

                if node_common_id is not None:
                    new_node = new_graph.get_node_by_id(node_common_id)
                elif node_dedup_id is not None:
                    new_node = new_graph.get_node_by_id(node_dedup_id)
                elif node.id in claimed_ids:
                    new_node, original_id = node.copy_and_refresh_id()
                    # TODO handle if node is generic
                    duplicate_id_old_to_new_lookup[original_id] = new_node.id
                else:
                    new_node = copy(node)
                    # TODO handle if node is generic

                new_graph.add_node(new_node)

        for edge in graph_1.get_edge_set():

            new_edge = copy(edge)

            ### FROM NODE SECTION ###
            from_node = PortGraph.__merge_graph_node_handler(new_graph, graph_1, common_ids_1_to_2,
                                                             duplicate_id_old_to_new_lookup, claimed_ids,
                                                             new_edge.source_node)
            new_edge.source_node = from_node.id

            ### TO NODE SECTION ###
            to_node = PortGraph.__merge_graph_node_handler(new_graph, graph_1, common_ids_1_to_2,
                                                           duplicate_id_old_to_new_lookup, claimed_ids,
                                                           new_edge.target_node)
            new_edge.target_node = to_node.id

            new_graph.add_existing_edge(from_node, to_node, new_edge)

        # Update generic count at the end
        new_graph.__generic_case_count = max(graph_1.get_generic_count(), graph_2.get_generic_count())

        return new_graph, duplicate_id_old_to_new_lookup


    # NOTE - SIDE EFFECTS ON THE ARG EDGE (IF THERE IS ONE) BUT RETURNED ANYWAY FOR OBVIOUS SIGNATURE
    @staticmethod
    def __merge_graph_node_handler(new_graph: 'PortGraph', original_graph: 'PortGraph',
                                   common_id_lookup: Dict[int, int],
                                   duplicate_id_lookup: Dict[int, int], claimed_ids: Set[int],
                                   original_node_id: int) -> Node:

        common_id: Optional[int] = common_id_lookup.get(original_node_id)
        node_dedup_id: Optional[int] = duplicate_id_lookup.get(original_node_id)

        # Case node is a prescribed common node
        if common_id is not None:
            new_node = new_graph.get_node_by_id(common_id)

        # Case node is an already-seen duplicate, unclaimed node that has been deduped already
        elif node_dedup_id is not None:
            new_node = new_graph.get_node_by_id(node_dedup_id)

        # Case node is duplicate, unclaimed node that has not been seen before
        elif original_node_id in claimed_ids:
            node = original_graph.get_node_by_id(original_node_id)
            new_node, original_id = node.copy_and_refresh_id()
            duplicate_id_lookup[original_id] = new_node.id
            if original_graph.is_node_generic(original_id):
                PortGraph.adopt_single_node_generics(new_graph, original_graph, duplicate_id_lookup, claimed_ids,
                                                     new_node, original_id)

        # Case node is unique, unclaimed
        else:
            node = original_graph.get_node_by_id(original_node_id)
            new_node = copy(node)  # This might be unnecessary but just being safe
            if original_graph.is_node_generic(original_node_id):
                PortGraph.adopt_single_node_generics(new_graph, original_graph, duplicate_id_lookup, claimed_ids,
                                                     new_node, original_node_id)

        return new_node


    @staticmethod
    def adopt_single_node_generics(new_graph, original_graph, duplicate_id_lookup, claimed_ids, new_node, original_id):
        original_concrete_node_ids = original_graph.__generic_node_lookup[original_id]
        if len(original_concrete_node_ids) == 0:
            raise Exception("Generic node has no concrete nodes")
        concrete_nodes = [original_graph.get_node_by_id(original_id, index)
                          for index in range(original_graph.get_generic_count())]
        concrete_nodes_deduped = [PortGraph.__merge_generic_concrete_node_handler(new_graph, original_graph,
                                                                                  duplicate_id_lookup, claimed_ids,
                                                                                  node.id)
                                  for node in concrete_nodes]
        # If the node is generic get:
        # 1. Reverse lookup entry (dict entry tuple[int] -> int)
        # 2. __generic_node_single_lookup sub dict (int (concrete) -> int (placeholder))
        # 3. Generic lookup entry (dict entry int -> tuple[int])
        # 4. generic (concrete) nodes by id sub dict
        concrete_tuple = tuple(node.id for node in concrete_nodes_deduped)
        # 1.
        # reverse_lookup_entry = (concrete_tuple, new_node.id)
        new_graph.__generic_node_reverse_lookup[concrete_tuple] = new_node.id
        # 2.
        generic_single_lookup_sub_dict = {node.id: new_node.id for node in concrete_nodes_deduped}
        new_graph.__generic_node_single_lookup.update(generic_single_lookup_sub_dict)
        # 3.
        # generic_lookup_entry = (new_node.id, concrete_tuple)
        new_graph.__generic_node_lookup[new_node.id] = concrete_tuple
        # 4.
        concrete_nodes_by_id_sub_dict = {node.id: node for node in concrete_nodes_deduped}
        new_graph.__generic_nodes_by_id.update(concrete_nodes_by_id_sub_dict)


    @staticmethod
    def __merge_generic_concrete_node_handler(new_graph: 'PortGraph', original_graph: 'PortGraph',
                                              duplicate_id_lookup: Dict[int, int], claimed_ids: Set[int],
                                              original_concrete_node_id: int) -> Node:

        # Dedup cases to handle:
        # 1. Common lookup INVALID
        # 2. Collision across graphs VALID
        # 3. Collision (already) across graphs VALID
        # 4. No collision VALID

        concrete_node_dedup_id = duplicate_id_lookup.get(original_concrete_node_id)

        if concrete_node_dedup_id is not None:
            new_concrete_node = new_graph.get_node_by_id(concrete_node_dedup_id)

        elif original_concrete_node_id in claimed_ids:
            concrete_node = original_graph.get_node_by_id(original_concrete_node_id)
            new_concrete_node, original_id = concrete_node.copy_and_refresh_id()
            duplicate_id_lookup[original_id] = new_concrete_node.id

        else:
            concrete_node = original_graph.get_node_by_id(original_concrete_node_id)
            new_concrete_node = copy(concrete_node)

        return new_concrete_node


