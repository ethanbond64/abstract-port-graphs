from collections import defaultdict
from typing import List, Set, Dict, Optional, Any

from port_graphs import PortGraph
from programs import Program
from nodes import DEFAULT_GROUP_BY, Constant, InputNode, InputSetNode, \
    DisjointSetNode, Node, OutputNode, OperatorNode, ProxyNode, RecursiveProxyNode, IterativeProxyNode, \
    RelationshipNode, SetJoin, SetSplit
from base_types import PerceivedType, DslSet, get_source_port_value
from interpreter.constraint_solver import RelationshipConstraintSolver
from interpreter.interpreter_graph_iteration import GraphInstanceIterator, IterationEvent, RecursionEvent, \
    IterativeProxyEvent, HaltInstanceEvent, JoinEvent, SplitEvent
from interpreter.interpreter_operator_plan import OperatorPlan
from interpreter.interpretter_common import GraphIO, traverse_connected_graph, GraphInstanceState, \
    get_args_immediate


class Interpreter:

    def __init__(self, initial_depth = 0):
        self.nested_function_depth = initial_depth

    def _in_nested_function(self):
        return self.nested_function_depth > 1

    def _enter_function(self):
        self.nested_function_depth += 1

    def _exit_function(self):
        self.nested_function_depth -= 1

    def evaluate_program(self, program: Program, raw_data: Any):

        perceived_objects = program.perception_model.apply_perception(raw_data)

        graph_io = self.populate_graph_io_inputs(program.graph, GraphIO(program.graph), perceived_objects)
        for connected_subgraph in get_connected_subgraphs(program.graph):
            graph_io = self.evaluate_port_graph(connected_subgraph, graph_io)

        return program.format_output_values(graph_io.get_output_values())


    @staticmethod
    def populate_graph_io_inputs(graph: PortGraph, graph_io: GraphIO,
                                 objects: List[PerceivedType]) -> GraphIO:

        for node in graph.get_nodes_by_id().values():
            if isinstance(node, InputNode):
                perception_qualifiers = node.perception_qualifiers
                filtered_instances = list(filter(lambda i: isinstance(i, node.get_outbound_type()) and
                                                           Interpreter.valid_perception_match(i, perception_qualifiers),
                                                 objects))
                graph_io.extend_input_values(node.id, filtered_instances)

            if isinstance(node, InputSetNode):
                group_by_dict = defaultdict(DslSet)
                for instance in objects:
                    if (isinstance(instance, node.member_type) and
                            Interpreter.valid_perception_match(instance, node.perception_qualifiers)):
                        group_key = get_source_port_value(instance, node.group_by) \
                            if node.group_by else DEFAULT_GROUP_BY
                        group_by_dict[group_key].add(instance)
                # Add all sets (values) to instances list
                graph_io.extend_input_values(node.id, group_by_dict.values())
        return graph_io

    @staticmethod
    def valid_perception_match(perceived_value: PerceivedType,
                               node_perception_qualifiers: Optional[Set[Any]]):

        if node_perception_qualifiers is None:
            return True

        return perceived_value.get_perception_function() in node_perception_qualifiers


    # Graph evaluated here must be connected.
    def evaluate_port_graph(self, graph: PortGraph, graph_io: GraphIO) -> GraphIO:

        try :
            self._enter_function()

            operator_plan = OperatorPlan.get_operator_plan(graph)
            root_nodes = [graph.get_node_by_id(root_node_id) for root_node_id in operator_plan.get_root_nodes()]
            constraint_solver = RelationshipConstraintSolver(graph_io, root_nodes, graph.get_generic_count(),
                                                             not self._in_nested_function())
            for relationship in operator_plan.get_immediate_relationships():
                relationship_node = graph.get_node_by_id(relationship)
                constraint_solver.add_relationship_constraint(graph, relationship_node)

            graph_instances = constraint_solver.get_graph_instances(graph)

            completed_graph_instances = self._run_operator_plan(graph_io, operator_plan, graph_instances)
            graph_io.collect_output_values(completed_graph_instances)

            return graph_io

        finally:
            self._exit_function()


    def _run_operator_plan(self, graph_io: GraphIO, operator_plan: OperatorPlan,
                           graph_instances: List[GraphInstanceState]):

        # Note eventually sort using runtime node... or re-sort if we recently added new graph instances.
        graph_instance_iterator = GraphInstanceIterator(operator_plan, graph_instances)

        while graph_instance_iterator.has_next():
            graph_instance, current_node_id = graph_instance_iterator.next()
            event = self._evaluate_single_node_instance(graph_instance, current_node_id)
            graph_instance_iterator.handle_iteration_event(graph_io, event)

        return graph_instance_iterator.get_instances()


    def _evaluate_single_node_instance(self, graph_instance: GraphInstanceState, node_id: int) -> Optional[IterationEvent]:

        args = get_args_immediate(graph_instance.graph, graph_instance, node_id)
        node = graph_instance.graph.get_node_by_id(node_id, graph_instance.generic_index)

        event = None # Event used to signal things like - recursion / joins / condition halts

        # Every "keyword" node in the grammar must have a case in this "switch".
        if isinstance(node, (InputNode, InputSetNode, OutputNode)):
            state = args[0]

        elif isinstance(node, Constant):
            state = node.value

        elif isinstance(node, OperatorNode):
            state = node.apply(*args) if node.primitive() else node.apply(self, *args)

        elif isinstance(node, ProxyNode) and args[1] is not None:
            # Case recursive value is being set (elsewhere), rather than determined for this instance.
            state = args[1]

            if isinstance(node, RecursiveProxyNode):
                if self.validate_safe_recursion(state):
                    event = RecursionEvent(node.id, graph_instance)
            elif isinstance(node, IterativeProxyNode):
                event = IterativeProxyEvent(node.id, state)
            else:
                # NOTE old behavior not implemented after cleanup
                # refer to commit fed28d8bd25e41b244e5f2185d34b863f5ba80db (last with old stages) to see
                # how rec-search was implemented
                raise NotImplementedError()

        elif isinstance(node, ProxyNode) and args[1] is None:
            # Case value to using in this graph instance is being determined.

            # If there is already a value in the graph state for the node we are evaluating, use it.
            # otherwise use the initial arg.
            existing_recursive_value = graph_instance.state.get(node_id)
            state = existing_recursive_value if existing_recursive_value is not None else args[0]

        elif isinstance(node, RelationshipNode):
            state = node.apply(*args)
            if not state:
                event = HaltInstanceEvent()

        # elif isinstance(node, SearchConditionNode):
        #     state = node.apply(*args)
        #     if state:
        #         event = IterationEvent.HALT_FUNCTION

        elif isinstance(node, (SetJoin, DisjointSetNode)):
            state = args[0]
            event = JoinEvent(node.id, isinstance(node, DisjointSetNode))

        elif isinstance(node, SetSplit):
            state = node.apply(*args)
            event = SplitEvent(node.id, state)

        else:
            print(node.__class__.__name__)
            raise Exception("Unimplemented keyword node type")

        # Set the state value in the graph state.
        graph_instance.state[node_id] = state

        return event

    # Intended to be overridden.
    def validate_safe_recursion(self, recursive_value: Any) -> bool:
        return True


# Breaks the full graph into connected subgraphs
def get_connected_subgraphs(graph: PortGraph) -> List[PortGraph]:

    all_seen_nodes = set()
    connected_node_groups: List[Set[Node]] = [] # Need to store the direction of each disjoint set
    # Indexes correspond to connected_node_groups above, values are dict {disjoint set id: boolean set is root}
    disjoint_set_directions: List[Dict[int, bool]] = []

    disjoint_set_node_ids = set(node.id for node in graph.get_nodes_by_type(DisjointSetNode.__name__))
    for node in graph.get_nodes_by_id().values():

        # Skip seen nodes, and disjoint sets must be discovered via graph traversal, not the initial search.
        if node.id in all_seen_nodes or node.id in disjoint_set_node_ids:
            continue

        visited = set()
        traverse_connected_graph(graph, node, visited)

        connected_node_groups.append(visited)
        disjoint_lookup = build_disjoint_set_lookup(graph, disjoint_set_node_ids, visited)
        disjoint_set_directions.append(disjoint_lookup)

        visited_sanitized = visited.difference(disjoint_set_node_ids)
        all_seen_nodes |= visited_sanitized

    subgraphs: List[PortGraph] = []
    for node_group_index, node_group in enumerate(connected_node_groups):

        if len(node_group) == 1:
            continue

        disjoint_set_lookup = disjoint_set_directions[node_group_index]

        subgraph = PortGraph()
        for node_id in node_group:
            node = graph.get_node_by_id(node_id)
            acquire_outbound_edges = disjoint_set_lookup.get(node.id, True)
            if acquire_outbound_edges:
                for existing_edge in graph.get_edges_from_by_id(node_id):
                    other_node = graph.get_node_by_id(existing_edge.target_node)
                    subgraph.add_existing_edge(node, other_node, existing_edge)

        if graph.has_generics():
            subgraph.adopt_generics(graph)

        subgraphs.append(subgraph)

    return subgraphs


def build_disjoint_set_lookup(complete_graph: PortGraph, disjoint_set_node_ids: Set[int],
                              visited: Set[int]) -> Dict[int, bool]:
    disjoint_set_lookup = {}
    current_disjoint_set_ids = visited.intersection(disjoint_set_node_ids)
    for disjoint_set_id in current_disjoint_set_ids:

        inbound_edge_node_seen = False
        for edge in complete_graph.get_edges_to_by_id(disjoint_set_id):
            if edge.source_node in visited:
                inbound_edge_node_seen = True
                break

        outbound_edge_node_seen = False
        for edge in complete_graph.get_edges_from_by_id(disjoint_set_id):
            if edge.target_node in visited:
                outbound_edge_node_seen = True
                break

        if inbound_edge_node_seen and outbound_edge_node_seen:
            raise Exception("Invalid disjoint set related visited subset.")

        if outbound_edge_node_seen:
            disjoint_set_lookup[disjoint_set_id] = True
        elif inbound_edge_node_seen:
            disjoint_set_lookup[disjoint_set_id] = False

    return disjoint_set_lookup
