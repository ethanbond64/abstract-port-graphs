from copy import copy
from typing import List, Tuple, Any

from base_types import DslSet
from interpreter.interpreter_operator_plan import OperatorPlan
from interpreter.interpretter_common import GraphIO, GraphInstanceState


class IterationEvent:
    ...


class HaltInstanceEvent(IterationEvent):
    ...


class IterativeProxyEvent(IterationEvent):
    def __init__(self, node_id: int, value):
        self.node_id = node_id
        self.value = value


class JoinEvent(IterationEvent):

    def __init__(self, join_node_id, disjoint_set = False):
        self.join_node_id = join_node_id
        self.disjoint_set = disjoint_set


class RecursionEvent(IterationEvent):
    def __init__(self, node_id: int, graph_instance: GraphInstanceState):
        self.node_id = node_id
        self.graph_instance = graph_instance


class SplitEvent(IterationEvent):
    def __init__(self, node_id, value_set: DslSet[Any]):
        self.node_id = node_id
        self.value_set = value_set


class GraphInstanceIterator:
    def __init__(self, operator_plan: OperatorPlan, initial_graph_instances: List[GraphInstanceState]):
        self.operator_nodes = operator_plan.get_non_constraint_nodes()
        self.operator_count = len(self.operator_nodes)
        self.join_indexes = list(sorted(self.operator_nodes.index(join_id) for join_id
                                        in operator_plan.get_join_nodes()))

        self.graph_instances = copy(initial_graph_instances)
        self.operator_matrix = [[] for _ in initial_graph_instances]
        self.instance_cursor = 0

        self.staged_join_indexes = set() # All prerequisites have run, but a successful join event has not yet.
        self.completed_join_indexes = set() # Join event completed.

        self.halted = (not self.graph_instances or not self.operator_nodes)

    def has_next(self) -> bool:
        return not (self.halted or
                    (self.__is_last_instance() and
                     self.__get_operator_index() == self.operator_count))

    def next(self) -> Tuple[GraphInstanceState, int]:

        increment_index = self.__get_instance_increment_index()
        if self.__get_operator_index() == increment_index:
            # Loop if we are on the last instance
            if self.__is_last_instance():
                self.instance_cursor = 0
            else:
                self.instance_cursor += 1

        operator_index = self.__get_operator_index()
        operator = self.operator_nodes[operator_index]
        self.operator_matrix[self.instance_cursor].append(operator)

        # If this is a join (increment index != operator count) and the final instance, stage the join (increment - 1)
        return_instance_cursor = self.instance_cursor
        if self.__is_last_instance() and increment_index != self.operator_count and self.__get_operator_index() == increment_index:
            self.staged_join_indexes.add(increment_index - 1)
            self.instance_cursor = 0

        return self.graph_instances[return_instance_cursor], operator

    def __is_last_instance(self) -> bool:
        return self.instance_cursor == len(self.operator_matrix) - 1

    def __get_operator_index(self) -> int:
        return len(self.operator_matrix[self.instance_cursor])

    def __get_instance_increment_index(self) -> int:
        # For each join node in the operator plan, determine which ones are still incomplete in any graph instances.
        # If there are any, return the join nodes operation index. Otherwise return the total number of operator nodes
        for join_node_index in self.join_indexes:
            if join_node_index not in self.staged_join_indexes and join_node_index not in self.completed_join_indexes:
                # Look for all completed except the last - this will be completed now...
                return join_node_index + 1

        return self.operator_count

    def handle_iteration_event(self, graph_state: GraphIO, event: IterationEvent):

        if isinstance(event, HaltInstanceEvent):
            self.graph_instances.pop(self.instance_cursor)
            self.operator_matrix.pop(self.instance_cursor)
            if self.instance_cursor == len(self.graph_instances):
                self.instance_cursor -= 1

        elif isinstance(event, IterativeProxyEvent):
            if not self.__is_last_instance():
                self.graph_instances[self.instance_cursor + 1].state[event.node_id] = event.value

        elif isinstance(event, JoinEvent):
            # Create a directive which will run every graph up to the join node,
            # then apply the join operation to all nodes
            event_node_index = self.operator_nodes.index(event.join_node_id)
            if event_node_index in self.staged_join_indexes and event_node_index not in self.completed_join_indexes:
                value_set = DslSet()
                # Collect all individual graph instance values.
                for graph_instance in self.graph_instances:
                    value_set.add(graph_instance.state[event.join_node_id])

                # Set a copy of the common collection on all instances.
                for graph_instance in self.graph_instances:
                    graph_instance.state[event.join_node_id] = copy(value_set)

                # If this is a disjoint set, store the set itself on the graph state,
                # so that a dependent disjoint subgraph can inherit the values in the future.
                if event.disjoint_set:
                    graph_state.set_disjoint_set(event.join_node_id, copy(value_set))

                self.completed_join_indexes.add(event_node_index)

        elif isinstance(event, RecursionEvent):
            copied_state = copy(event.graph_instance.state)
            recursive_node_first_index = self.operator_nodes.index(event.node_id)
            for operator_to_clear in self.operator_nodes[recursive_node_first_index + 1:]:
                if operator_to_clear != event.node_id:
                    del copied_state[operator_to_clear]

            new_instance = GraphInstanceState(event.graph_instance.graph, copied_state, event.graph_instance.generic_index)
            new_instance.parent_id = event.graph_instance.id

            self.graph_instances.append(new_instance)
            self.operator_matrix.append([op for op in self.operator_nodes[:recursive_node_first_index]])

        elif isinstance(event, SplitEvent):
            # NOTE this impl assumes the set split was not the last operator evaluated.
            if event.value_set:
                current_instance = self.graph_instances.pop(self.instance_cursor)
                current_operator_list = self.operator_matrix.pop(self.instance_cursor)

                copied_instances = []
                copied_operator_lists = []
                for value in event.value_set:
                    copied_state = copy(current_instance.state)
                    copied_state[event.node_id] = value
                    new_instance = GraphInstanceState(current_instance.graph, copied_state, current_instance.generic_index)
                    new_instance.parent_id = current_instance.id
                    copied_instances.append(new_instance)
                    copied_operator_lists.append(copy(current_operator_list))

                self.graph_instances[self.instance_cursor:self.instance_cursor] = copied_instances
                self.operator_matrix[self.instance_cursor:self.instance_cursor] = copied_operator_lists

    def get_instances(self) -> List[GraphInstanceState]:
        return self.graph_instances
