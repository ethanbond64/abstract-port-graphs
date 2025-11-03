from typing import Dict, List, Iterable, Optional

from constraint import Problem, AllDifferentConstraint
from nodes import RelationshipNode, Constant, InputNode, OutputNode, \
    InputSetNode, DisjointSetNode, Node, GenericBaseNode
from port_graphs import Edge, GenericEdge, PortGraph
from interpreter.interpretter_common import GraphIO, GraphInstanceState
from base_types import get_source_port_value

GENERIC_KEY_INDEX = -1

class RelationshipConstraintSolver:

    def __init__(self, graph_state: GraphIO, starter_nodes: List[Node], generic_count: int,
                 require_all_different = True):
        self.problem = Problem()
        self.objects_by_id = {}
        self.constant_values_by_id = {}
        self.starter_nodes = starter_nodes

        # Add all variables
        for starter in starter_nodes:
            starter_instances = graph_state.get_input_values(starter.id)

            if isinstance(starter, InputNode):
                object_ids = []
                for instance in starter_instances:
                    object_id = id(instance)
                    object_ids.append(object_id)
                    self.objects_by_id[object_id] = instance
                self.problem.addVariable(starter.id, object_ids)

            elif isinstance(starter, InputSetNode):
                set_variables = []
                for index, instance in enumerate(starter_instances):
                    # Sets don't have an id, so make a tuple out of the node id and the index.
                    var_id = (starter.id, index)
                    self.objects_by_id[var_id] = instance
                    set_variables.append(var_id)
                self.problem.addVariable(starter.id, set_variables)

            elif isinstance(starter, Constant):
                self.constant_values_by_id[starter.id] = starter.value

            elif isinstance(starter, DisjointSetNode):
                object_id = (starter.id,)
                self.objects_by_id[object_id] = graph_state.get_disjoint_set(starter.id)
                self.problem.addVariable(starter.id, [object_id])

            else:
                raise TypeError("Unexpected starter node type: " + starter.__class__.__name__)

        if generic_count > 0:
            # Wrap the generic variable -1's values in a tuple with "g" so it never collides on the unique constraint.
            self.problem.addVariable(GENERIC_KEY_INDEX, [("g", i) for i in range(generic_count)])

        # Distinct constraint always holds for outer graphs, but not function node graphs
        if require_all_different:
            self.problem.addConstraint(AllDifferentConstraint())

    def add_relationship_constraint(self, graph: PortGraph, relationship: RelationshipNode):

        input_edges = graph.get_edges_to_by_id(relationship.id)

        if graph.has_generics():
            self.__add_generic_relationship_constraint(graph, relationship, input_edges)
            # NOTE: short circuit return
            return

        # Sort by target port, if constant put value in arg list, else put none and track signature index to arglist index
        arg_list = []
        signature_index_to_port = []
        signature_index_to_arg_index = []
        signature_variable_names = []

        for idx, input_edge in enumerate(sorted(input_edges, key=lambda e: e.target_port)):
            input_node = graph.get_node_by_id(input_edge.source_node)
            if isinstance(input_node, Constant):
                value = get_source_port_value(input_node.value, input_edge.source_port)
                arg_list.append(value)
            elif isinstance(input_node, (InputNode, OutputNode, InputSetNode, DisjointSetNode)):
                signature_index_to_port.append(input_edge.source_port)
                signature_index_to_arg_index.append(idx)
                signature_variable_names.append(input_node.id)
                arg_list.append(None)
            else:
                raise Exception("Node type unexpected")

        def constraint_function(*args):
            iterable_args = args
            for i, arg in enumerate(iterable_args):
                obj = self.objects_by_id[arg]
                port = signature_index_to_port[i]
                val = get_source_port_value(obj, port)
                arg_index = signature_index_to_arg_index[i]
                arg_list[arg_index] = val

            return relationship.apply(*arg_list)

        self.problem.addConstraint(constraint_function, signature_variable_names)

    def __add_generic_relationship_constraint(self, graph: PortGraph, relationship: RelationshipNode, edges: Iterable[Edge]):

        signature_variable_names = []

        arg_edges = []
        constant_ids_and_edges = []

        for edge in edges:
            # NOTE SOURCE NODE CANNOT BE GENERIC FOR NOW
            source_node = graph.get_node_by_id(edge.source_node)

            if isinstance(source_node, Constant):
                constant_ids_and_edges.append((source_node.id, edge))
                continue

            # Non-constant generics cannot be source nodes
            if isinstance(source_node, GenericBaseNode):
                raise Exception("Generic source node not supported")

            arg_edges.append(edge)
            signature_variable_names.append(source_node.id)

        signature_variable_names.append(GENERIC_KEY_INDEX)

        # NOTE GENERIC CONSTRAINT FUNCTION IS MUCH LESS EFFICIENT.
        # ALL NODES INCLUDING CONSTANTS DERIVED AT RUNTIME.
        def constraint_function(*args):
            generic_index = args[-1][1]
            effective_args = args[:-1]

            # Tuple of value and target port (to be sorted once all values in the list)
            relationship_args = []

            for arg_index, arg in enumerate(effective_args):
                obj = self.objects_by_id[arg]
                generic_edge = arg_edges[arg_index]
                concrete_edge = generic_edge.concrete(generic_index) if isinstance(generic_edge, GenericEdge) else generic_edge
                val = get_source_port_value(obj, concrete_edge.source_port)
                relationship_args.append((val, concrete_edge.target_port))

            for constant_id, constant_edge in constant_ids_and_edges:
                constant_node: Constant = graph.get_node_by_id(constant_id, generic_index)
                concrete_edge = constant_edge.concrete(generic_index) if isinstance(constant_edge, GenericEdge) else constant_edge
                val = get_source_port_value(constant_node.value, concrete_edge.source_port)
                relationship_args.append((val, concrete_edge.target_port))

            relationship_args = map(lambda tup: tup[0], sorted(relationship_args, key=lambda tup: tup[1]))
            concrete_relationship: RelationshipNode = graph.get_node_by_id(relationship.id, generic_index)

            relationship_args = list(relationship_args)

            return concrete_relationship.apply(*relationship_args)

        self.problem.addConstraint(constraint_function, signature_variable_names)

    def get_graph_instances(self, graph: PortGraph) -> List[GraphInstanceState]:

        # Substitute objects
        solutions = self.problem.getSolutions()
        if len(solutions) == 0 and len(self.objects_by_id) == 0:
            solutions = [{}]

        instances = []
        for d in solutions:
            generic_index = RelationshipConstraintSolver.__pop_generic_index(d)
            instances.append(GraphInstanceState(graph, self.__prep_dict_for_return(graph, d, generic_index), generic_index))

        return instances

    @staticmethod
    def __pop_generic_index(d: Dict):
        if GENERIC_KEY_INDEX not in d:
            return None
        return d.pop(GENERIC_KEY_INDEX)[1]

    def __prep_dict_for_return(self, graph: PortGraph, d: Dict, generic_index: Optional[int]):
        for key, value in d.items():
            d[key] = self.objects_by_id[value]

        generic_safe_constants = {k: (graph.get_node_by_id(k, generic_index).value if v is None else v)
                                  for k, v in self.constant_values_by_id.items()}

        d.update(generic_safe_constants)
        return d