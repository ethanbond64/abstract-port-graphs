import uuid
from collections import defaultdict
from copy import deepcopy, copy
from typing import Tuple, List, Type, DefaultDict, Optional, Dict, Any, Callable, Union

from arc.arc_objects import CENTERED, LocationValue
from nodes import Constant, OperatorNode, RelationshipNode, Node, InputNode
from port_graphs import Edge, PortGraph
from operator_primitives import ConstructorOperator
from base_types import get_source_port_value
from synthesis.arc_specific.catalog_class_structure import CLASS_COMPONENTS_CATALOG_FLATTENED, CLASS_CONSTRUCTOR_CATALOG
from synthesis.arc_specific.catalog_operators import OPERATOR_TO_STRATEGY
from synthesis.arc_specific.catalog_relationships_new import RelationshipSearchStrategy, TEMP_RELATIONSHIP_SEARCH_STRATEGIES
from synthesis.design_time_models import WildCardNode
from synthesis.program_state import ProgramTrace, TraceGraphInstance
from synthesis.program_state import ProgramState, EvaluationContext
from synthesis.synthesis_models import ExtendUpEdgeFormula, ExtendUpNodeFormula, TraceInstanceMapping, ExtendUpEdgeKey, \
    ExtendUpNodeKey, ForkJoinKey, LeafJoinKey, DirectiveType, ExtendDownConstructorKey, OperatorInstanceFormula, \
    ExtendDownConstantKey, InputInstanceFormula, ExtendDownInputKey, ExtendDownOperatorKey, BasicSynthesisActions, \
    RefocusPredicateKey


class SynthesisDirective:

    def __init__(self, directive_type: DirectiveType):
        self.__directive_type = directive_type

    def get_directive_type(self) -> DirectiveType:
        return self.__directive_type

    def run(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        raise NotImplementedError()

    def get_parameterized_trace_ids(self) -> Tuple[int, ...]:
        raise NotImplementedError()


class ExtensionSynthesisDirective(SynthesisDirective):

    def __init__(self, directive_type: DirectiveType, focus_trace: ProgramTrace,
                 next_dsl_member: Union[Type[Node], Type[Edge]]):
        super().__init__(directive_type)
        self.focus_trace: ProgramTrace = focus_trace
        self.focus_node = focus_trace.graph.get_node_by_id(focus_trace.focus_node_id)
        self.next_dsl_member: Type[Node] = next_dsl_member

    def _run_internal(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        raise NotImplementedError()

    def run(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        results = self._run_internal(program_state, evaluation_context)
        self.focus_trace.record_directive(self)
        action_type = BasicSynthesisActions.EXTEND_UP if self.focus_trace.search_direction_forward else BasicSynthesisActions.EXTEND_DOWN
        program_state.get_trace_graph().record_directive(self.focus_trace.id, action_type, self.next_dsl_member, None, results)
        return results

    def get_parameterized_trace_ids(self) -> Tuple[int, ...]:
        return (self.focus_trace.id,)


class ExtendDownConstantDirective(ExtensionSynthesisDirective):

    def __init__(self, focus_trace: ProgramTrace):
        super().__init__(DirectiveType.EXTEND_DOWN_CONSTANT, focus_trace, Constant)

    def _run_internal(self, program_state: 'ProgramState', evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        # Constant graphs (make one per instance value)
        values_to_sub_instances: TraceInstanceMapping[ExtendDownConstantKey, TraceGraphInstance] = self.__extend_down_constants_keyed_instances(program_state, evaluation_context)
        collected_constant_traces = []

        for directive_key, sub_instances in values_to_sub_instances.items():
            value_type = directive_key.value_type
            value = directive_key.value

            # TODO need a domain agnostic way to apply this sanitization.
            if value_type == LocationValue and value == CENTERED:
                value = copy(CENTERED)
                directive_key = ExtendDownConstantKey(value_type, value)

            constant_trace = program_state.get_trace_graph().get_or_create_traces(
                (self.focus_trace.id,), directive_key,
                lambda: [self.__extend_down_constants_trace_function(program_state, value_type, value)]
            )[0]
            collected_constant_traces.append(constant_trace)
            constant_trace.update_instances(program_state, sub_instances, evaluation_context)

        return collected_constant_traces

    def __extend_down_constants_keyed_instances(self, program_state: ProgramState, evaluation_context: EvaluationContext) -> TraceInstanceMapping[ExtendDownConstantKey, TraceGraphInstance]:
        values_to_sub_instances: TraceInstanceMapping[ExtendDownConstantKey, TraceGraphInstance] = TraceInstanceMapping()
        for original_id, instances in self.focus_trace.graph_instances(program_state, evaluation_context).items():
            for instance in instances:
                value = instance.state[self.focus_node.id]
                directive_key = ExtendDownConstantKey(self.focus_node.value_type, value)
                sub_instances = values_to_sub_instances[directive_key]
                instance_copy = copy(instance)
                if original_id in sub_instances:
                    sub_instances[original_id].append(instance_copy)
                else:
                    sub_instances[original_id] = [instance_copy]

        return values_to_sub_instances

    def __extend_down_constants_trace_function(self, program_state: ProgramState,
                                             value_type: Type,
                                             value: Any) -> ProgramTrace:
        new_graph = deepcopy(self.focus_trace.graph)
        new_constant = Constant(value_type, value)
        new_graph.replace_node(self.focus_node, new_constant)
        terminal = True  # len(new_graph.get_nodes_by_type(WildCardNode2.__name__)) == 0
        new_trace = ProgramTrace([self.focus_trace], new_graph, new_constant.id, self.focus_trace.depth + 1, terminal, self.focus_trace.latest_fork,
                                    self.focus_trace.latest_fork_index, self.focus_trace.root_value_type, program_state)
        # NOTE THIS ONLY APPLIES TO TERMINALS
        program_state.get_trace_graph().record_concretized_branch(self.focus_trace.id, new_trace.id)
        return new_trace


class ExtendDownInputDirective(ExtensionSynthesisDirective):

    def __init__(self, focus_trace: ProgramTrace):
        super().__init__(DirectiveType.EXTEND_DOWN_INPUT, focus_trace, InputNode)

    def _run_internal(self, program_state: ProgramState, evaluation_context: EvaluationContext) -> List[ProgramTrace]:

        input_instance_mapping = self.__extend_down_inputs_keyed_instances(program_state, evaluation_context)
        collected_input_traces = []

        for directive_key, sub_instance_dict in input_instance_mapping.items():

            input_key = (directive_key.input_type, directive_key.existing_node_id, directive_key.path)
            input_trace = (program_state.get_trace_graph()
                           .get_or_create_traces((self.focus_trace.id,), directive_key,
                                                 lambda: [self.__extend_down_inputs_trace_function(program_state, input_key)]))[0]
            collected_input_traces.append(input_trace)

            # Update the subinstances with node id from the trace before
            complete_instances: Dict[int, List[TraceGraphInstance]] = defaultdict(list)
            for og_perceived_id, sub_instance_tuples in sub_instance_dict.items():
                for sub_instance_formula in sub_instance_tuples:
                    sub_instance = sub_instance_formula.to_instance(input_trace.focus_node_id)
                    complete_instances[og_perceived_id].append(sub_instance)

            input_trace.update_instances(program_state, complete_instances, evaluation_context)

        return collected_input_traces

    def __extend_down_inputs_trace_function(self, program_state: ProgramState, input_key: Tuple[Type, Optional[int], Optional[str]]) -> ProgramTrace:

        # Create input node up here so it can be referenced in concrete to abstract lookups in the graph instances
        new_graph = deepcopy(self.focus_trace.graph)
        node_type, existing_node_id, path = input_key
        if existing_node_id is None:
            input_node = InputNode(value_type=node_type, perception_qualifiers=None)
            new_graph.replace_node(self.focus_node, input_node, source_port=path)
        else:
            input_node = new_graph.get_node_by_id(existing_node_id)
            new_graph.replace_node_with_existing_node(self.focus_node, input_node, source_port=path)

        terminal = True
        new_trace = ProgramTrace([self.focus_trace], new_graph, input_node.id, self.focus_trace.depth + 1, terminal, self.focus_trace.latest_fork,
                                    self.focus_trace.latest_fork_index, self.focus_trace.root_value_type, program_state)
        # NOTE THIS ONLY APPLIES TO TERMINALS
        program_state.get_trace_graph().record_concretized_branch(self.focus_trace.id, new_trace.id)
        return new_trace

    # Fork join should also only deal with input nodes under the focus node being joined on
    def __extend_down_inputs_keyed_instances(self, program_state: ProgramState,
                                           evaluation_context: EvaluationContext) -> TraceInstanceMapping[ExtendDownInputKey, InputInstanceFormula]:
        mapping: TraceInstanceMapping[ExtendDownInputKey, InputInstanceFormula] = TraceInstanceMapping()
        for original_perceived_id, instances in self.focus_trace.graph_instances(program_state, evaluation_context).items():
            case_index = program_state.get_perception_xref().get_perceived_value(original_perceived_id).case_index
            for instance in instances:

                value = instance.state[self.focus_node.id]
                for input_type, input_id, attribute_path in (program_state.get_uvn()
                        .get_input_attribute_paths(self.focus_node.value_type, value, case_index)):
                    # If its a predicate, then the original perceived id came from the input side
                    if evaluation_context.predicate_mode and input_id == original_perceived_id:
                        continue
                    existing_abstract_id = instance.get_abstract_from_concrete(input_id)
                    directive_key = ExtendDownInputKey(input_type, existing_abstract_id, attribute_path)
                    sub_instances_tuples = mapping[directive_key]
                    instance_copy = copy(instance)
                    instance_formula = InputInstanceFormula(input_id, instance_copy)
                    if original_perceived_id in sub_instances_tuples:
                        sub_instances_tuples[original_perceived_id].append(instance_formula)
                    else:
                        sub_instances_tuples[original_perceived_id] = [instance_formula]
        return mapping


class ExtendDownConstructorDirective(ExtensionSynthesisDirective):

    def __init__(self, focus_trace: ProgramTrace):
        super().__init__(DirectiveType.EXTEND_DOWN_CONSTRUCTOR, focus_trace, ConstructorOperator)

    def _run_internal(self, program_state: 'ProgramState', evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        # Single graph, but each wildcard contributor will have different instances based on the current wildcard value
        # Outer key is: (Value Type, int constructor index (for multiple constructors))
        mapping = self.__extend_down_constructor_keyed_instances(program_state, evaluation_context)
        collected_arg_traces = []

        for directive_key, sub_instance_dict in mapping.items():
            trace_list = (program_state.get_trace_graph().get_or_create_traces((self.focus_trace.id,), directive_key,
                          lambda: self.__tuple_to_list_utility(self.__extend_down_constructor_traces_function(program_state,
                                                                              directive_key.constructor_index))))

            constructor_trace, arg_traces = trace_list[0], trace_list[1:]
            collected_arg_traces.extend(trace_list)

            arg_node_ids = [n.focus_node_id for n in arg_traces]

            complete_instances: Dict[int, List[TraceGraphInstance]] = defaultdict(list)
            for original_id, instance_formulas in sub_instance_dict.items():
                for instance_formula in instance_formulas:
                    instance = instance_formula.to_instance(constructor_trace.focus_node_id, arg_node_ids)
                    complete_instances[original_id].append(instance)

            constructor_trace.update_instances(program_state, complete_instances, evaluation_context)
            for arg_trace in arg_traces:
                arg_trace.update_instances(program_state, complete_instances, evaluation_context)

        return collected_arg_traces

    @staticmethod
    def __tuple_to_list_utility(trace_tuple: Tuple[ProgramTrace, List[ProgramTrace]]) -> List[ProgramTrace]:
        return [trace_tuple[0]] + trace_tuple[1]

    def __extend_down_constructor_traces_function(self, program_state: ProgramState, constructor_index: int) -> Tuple[ProgramTrace, List[ProgramTrace]]:

        # Build graph first to get new node ids
        constructor_metadata = CLASS_CONSTRUCTOR_CATALOG.get(self.focus_node.value_type)[constructor_index]
        new_graph = deepcopy(self.focus_trace.graph)
        constructor_node = ConstructorOperator()
        new_graph.replace_node(self.focus_node, constructor_node)

        type_node = Constant(Type, constructor_metadata.constructor)
        new_graph.add_edge(type_node, constructor_node, to_port=0)

        i = 1
        arg_node_ids = []
        for component_path, component_type in constructor_metadata.components:
            new_wildcard_node = WildCardNode(component_type)
            arg_node_ids.append(new_wildcard_node.id)
            new_graph.add_edge(new_wildcard_node, constructor_node, to_port=i)
            i += 1

        terminal = False

        # NOTE create a trace for each wildcard arg - 1-1 so each gets a separate focus
        fork_trace = ProgramTrace([self.focus_trace], new_graph, constructor_node.id, self.focus_trace.depth, terminal,
                                  self.focus_trace.latest_fork, self.focus_trace.latest_fork_index, self.focus_trace.root_value_type,
                                  program_state)

        # NOTE THEY ARE ALL USING THE SAME COLLECTIONS - THIS MAY CAUSE PROBLEMS
        arg_traces = []
        for i, arg_node_id in enumerate(arg_node_ids):
            arg_trace = ProgramTrace([fork_trace], new_graph, arg_node_id, self.focus_trace.depth + 1, terminal, fork_trace.id,
                                     i + 1, self.focus_trace.root_value_type, program_state)
            arg_traces.append(arg_trace)

        return fork_trace, arg_traces

    def __extend_down_constructor_keyed_instances(self, program_state: ProgramState,
                                                evaluation_context: EvaluationContext) -> TraceInstanceMapping[ExtendDownConstructorKey, OperatorInstanceFormula]:
        mapping: TraceInstanceMapping[ExtendDownConstructorKey, OperatorInstanceFormula] = TraceInstanceMapping()
        for original_id, instances in self.focus_trace.graph_instances(program_state, evaluation_context).items():
            for instance in instances:
                cons_value = instance.state[self.focus_node.id]
                constructors = CLASS_CONSTRUCTOR_CATALOG.get(self.focus_node.value_type, [])
                for constructor_index, constructor_metadata in enumerate(constructors):
                    if constructor_metadata.is_applicable(cons_value):
                        arg_values: List[Any] = list(
                            map(lambda cmp: (cmp[1], get_source_port_value(cons_value, cmp[0])), constructor_metadata.components))
                        directive_key = ExtendDownConstructorKey(constructor_index)
                        components_and_sub_instances = mapping[directive_key]
                        instance_copy = copy(instance)
                        instance_formula = OperatorInstanceFormula(arg_values, instance_copy)
                        if original_id in components_and_sub_instances:
                            components_and_sub_instances[original_id].append(instance_formula)
                        else:
                            components_and_sub_instances[original_id] = [instance_formula]
        return mapping


class ExtendDownOperatorDirective(ExtensionSynthesisDirective):

    def __init__(self, focus_trace: ProgramTrace, operator_node_type: Type[OperatorNode]):
        super().__init__(DirectiveType.EXTEND_DOWN_OPERATOR, focus_trace, operator_node_type)

    def _run_internal(self, program_state: 'ProgramState', evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:

        mapping: TraceInstanceMapping[ExtendDownOperatorKey, OperatorInstanceFormula] = self.__extend_down_operators_keyed_instances(program_state, evaluation_context)
        collected_arg_traces = []

        for directive_key, sub_instance_dict in mapping.items():
            trace_list = (program_state.get_trace_graph()
                          .get_or_create_traces((self.focus_trace.id,), directive_key,
                          lambda: self.__extend_down_operators_trace_function(program_state,
                                                                              directive_key.operator)))

            operator_trace, arg_traces = trace_list[0], trace_list[1:]
            collected_arg_traces.extend(trace_list)
            arg_node_ids = [n.focus_node_id for n in arg_traces]

            complete_instances: Dict[int, List[TraceGraphInstance]] = defaultdict(list)
            for original_id, instance_formulas in sub_instance_dict.items():
                for instance_formula in instance_formulas:
                    instance = instance_formula.to_instance(operator_trace.focus_node_id, arg_node_ids)
                    complete_instances[original_id].append(instance)

            operator_trace.update_instances(program_state, complete_instances, evaluation_context)
            for arg_trace in arg_traces:
                arg_trace.update_instances(program_state, complete_instances, evaluation_context)

        return collected_arg_traces

    def __extend_down_operators_trace_function(self, program_state: ProgramState,
                                             operator_template: OperatorNode) -> Tuple[ProgramTrace, List[ProgramTrace]]:

        new_graph = deepcopy(self.focus_trace.graph)
        operator_node = operator_template.copy_and_refresh_id()[0]
        new_graph.replace_node(self.focus_node, operator_node)

        arg_wildcard_ids = []

        for i, input_type in enumerate(operator_template.input_types):
            new_wildcard = WildCardNode(input_type)
            new_graph.add_edge(new_wildcard, operator_node, to_port=i)
            arg_wildcard_ids.append(new_wildcard.id)

        terminal = False

        # NOTE create a trace for each wildcard arg - 1-1 so each gets a separate focus
        # NOTE THEY ARE ALL USING THE SAME COLLECTIONS - THIS MAY CAUSE PROBLEMS
        fork_trace = ProgramTrace([self.focus_trace], new_graph, operator_node.id, self.focus_trace.depth, terminal, self.focus_trace.latest_fork,
                                  self.focus_trace.latest_fork_index, self.focus_trace.root_value_type, program_state)

        arg_traces = []
        for i, arg_node_id in enumerate(arg_wildcard_ids):
            arg_trace = ProgramTrace([fork_trace], new_graph, arg_node_id, self.focus_trace.depth + 1, terminal, fork_trace.id, i,
                                     self.focus_trace.root_value_type, program_state)
            arg_traces.append(arg_trace)

        return [fork_trace] + arg_traces

    def __extend_down_operators_keyed_instances(self, program_state: ProgramState, evaluation_context: EvaluationContext) -> TraceInstanceMapping[ExtendDownOperatorKey, OperatorInstanceFormula]:
        mapping: TraceInstanceMapping[ExtendDownOperatorKey, OperatorInstanceFormula] = TraceInstanceMapping()
        # for potential_operator in TYPE_TO_OPERATOR[self.focus_node.value_type]:
        potential_operator_class: Type[OperatorNode] = self.next_dsl_member

        # TODO temp until dict is migrated to hold classes not instances
        potential_operator = None
        strategy = None
        for op, strat in OPERATOR_TO_STRATEGY.items():
            if op.__class__ == potential_operator_class:
                potential_operator = op
                strategy = strat

        if strategy is None:
            raise NotImplementedError()

        directive_key = ExtendDownOperatorKey(potential_operator)
        for original_id, out_graph_instances in self.focus_trace.graph_instances(program_state, evaluation_context).items():
            case_index = program_state.get_perception_xref().get_perceived_value(original_id).case_index
            for out_graph_instance in out_graph_instances:
                value = out_graph_instance.state[self.focus_node.id]
                for arg_combo in strategy.generate_argument_value_tuples(program_state.get_uvn(), self.focus_node.value_type,
                                                                         value, case_index):

                    # TEMP TO STOP RECURSION UNTIL ALLOWANCE IS MADE!
                    if any(arg_tup == (self.focus_node.value_type, value) for arg_tup in arg_combo):
                        continue

                    instance_copy = copy(out_graph_instance)
                    instance_formula = OperatorInstanceFormula(arg_combo, instance_copy)
                    sub_instances = mapping[directive_key]

                    if original_id in sub_instances:
                        sub_instances[original_id].append(instance_formula)
                    else:
                        sub_instances[original_id] = [instance_formula]
        return mapping


class ExtendUpEdgeDirective(ExtensionSynthesisDirective):

    def __init__(self, focus_trace: ProgramTrace):
        super().__init__(DirectiveType.EXTEND_UP_EDGE, focus_trace, Edge)

    def _run_internal(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        focus_condition = self.focus_trace
        focus_node = self.focus_node
        collected_traces = []

        mapping = self.__extend_up_edge_keyed_instances(program_state, evaluation_context, focus_condition, focus_node)

        for directive_key, sub_instances in mapping.items():
            sub_tuple = (directive_key.sub_path, directive_key.sub_type)
            new_trace = program_state.get_trace_graph().get_or_create_traces(
                (focus_condition.id,), directive_key,
                lambda: [self.__extend_up_edge_trace_function(program_state, focus_condition, sub_tuple)]
            )[0]
            collected_traces.append(new_trace)

            complete_instances: Dict[int, List[TraceGraphInstance]] = defaultdict(list)
            for original_id, instance_formulas in sub_instances.items():
                for instance_formula in instance_formulas:
                    instance = instance_formula.to_instance(new_trace.focus_node_id)
                    complete_instances[original_id].append(instance)

            new_trace.update_instances(program_state, complete_instances, evaluation_context)

        return collected_traces

    def __extend_up_edge_keyed_instances(self, program_state: ProgramState, evaluation_context: EvaluationContext, focus_trace: ProgramTrace,
                                         focus_node: Node) -> TraceInstanceMapping[ExtendUpEdgeKey, 'ExtendUpEdgeFormula']:

        mapping: TraceInstanceMapping[ExtendUpEdgeKey, ExtendUpEdgeFormula] = TraceInstanceMapping()
        for sub_path, sub_type in CLASS_COMPONENTS_CATALOG_FLATTENED[focus_node.get_outbound_type()]:
        # for sub_path, sub_type in ARC_LIBRARY.get_flattened_components(focus_node.get_outbound_type()):
            directive_key = ExtendUpEdgeKey(sub_path, sub_type)
            sub_instances: Dict[int, List[ExtendUpEdgeFormula]] = mapping[directive_key]
            for perceived_input_id, instances in focus_trace.graph_instances(program_state, evaluation_context).items():
                for instance in instances:
                    copied_instance = copy(instance)
                    instance_formula = ExtendUpEdgeFormula(focus_trace.focus_node_id, sub_path, copied_instance)
                    if perceived_input_id in sub_instances:
                        sub_instances[perceived_input_id].append(instance_formula)
                    else:
                        sub_instances[perceived_input_id] = [instance_formula]
        return mapping

    def __extend_up_edge_trace_function(self, program_state: ProgramState, parent_trace: ProgramTrace,
                                      sub_tuple: Tuple[Optional[str], Type]) -> ProgramTrace:

        sub_path, sub_type = sub_tuple
        new_graph = deepcopy(parent_trace.graph)
        source_node = new_graph.get_node_by_id(parent_trace.focus_node_id)
        new_focus_node = WildCardNode(sub_type)
        new_graph.add_edge(source_node, new_focus_node, from_port=sub_path)

        return ProgramTrace([parent_trace], new_graph, new_focus_node.id, parent_trace.depth, False, None, None,
                                    bool, program_state, search_direction_forward=True)


class ExtendUpNodeDirective(ExtensionSynthesisDirective):

    # TODO relationship arg
    def __init__(self, focus_trace: ProgramTrace, relationship_or_operator: Union[Type[RelationshipNode], Type[OperatorNode]] = RelationshipNode):
        super().__init__(DirectiveType.EXTEND_UP_NODE, focus_trace, relationship_or_operator) # TODO
        self._temp_cache__new_relationship = None
        self._temp_cache__port_to_arg_traces = None

    def _run_internal(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        focus_condition = self.focus_trace
        focus_node = self.focus_node
        collected_arg_traces = []

        if focus_condition.wildcard_count == 0:
            raise Exception("Invalid assumption")

        if not isinstance(focus_node, WildCardNode):
            raise Exception("Invalid assumption")

        mapping: TraceInstanceMapping[ExtendUpNodeKey, 'ExtendUpNodeFormula'] = self.__extend_up_node_keyed_instances(program_state, evaluation_context,
                                                                                                                      focus_condition, focus_node)
        for directive_key, sub_instances in mapping.items():

            program_state.get_trace_graph().get_or_create_traces(
                (focus_condition.id,), directive_key,
                lambda: self.__extend_node_up_trace_function(program_state, focus_condition, focus_node, directive_key.relationship_strategy)
            )
            new_relationship_node = self._temp_cache__new_relationship
            port_to_arg_traces = self._temp_cache__port_to_arg_traces
            collected_arg_traces.extend(v for v in port_to_arg_traces.values())
            port_to_arg_node_ids = {port_index: arg_trace.focus_node_id for port_index, arg_trace in port_to_arg_traces.items()}

            complete_instances: Dict[int, List[TraceGraphInstance]] = defaultdict(list)
            for original_id, instance_formulas in sub_instances.items():
                for instance_formula in instance_formulas:
                    instance = instance_formula.to_instance(new_relationship_node.id, port_to_arg_node_ids)
                    complete_instances[original_id].append(instance)

            for arg_trace in port_to_arg_traces.values():
                arg_trace.update_instances(program_state, complete_instances, evaluation_context)

        return collected_arg_traces

    def __extend_up_node_keyed_instances(self, program_state: ProgramState, evaluation_context: EvaluationContext, focus_trace: ProgramTrace,
                                         focus_node: WildCardNode) -> TraceInstanceMapping[ExtendUpNodeKey, 'ExtendUpNodeFormula']:

        mapping = TraceInstanceMapping()
        all_relationship_strategies: List[RelationshipSearchStrategy] = TEMP_RELATIONSHIP_SEARCH_STRATEGIES[Any] + \
                                                                        TEMP_RELATIONSHIP_SEARCH_STRATEGIES[
                                                                            focus_node.value_type]
        for relationship_strategy in all_relationship_strategies:

            directive_key = ExtendUpNodeKey(relationship_strategy)
            sub_instances = mapping[directive_key]

            for perceived_input_id, instances in focus_trace.graph_instances(program_state, evaluation_context).items():
                for instance in instances:

                    case_index = program_state.get_perception_xref().get_perceived_value(perceived_input_id).case_index
                    for ports_to_arg_tuples in (relationship_strategy
                            .generate_other_argument_tuples(program_state.get_uvn(), focus_node.value_type, instance.state[focus_node.id], case_index)):
                        instance_copy = copy(instance)
                        instance_formula = ExtendUpNodeFormula(ports_to_arg_tuples, instance_copy)
                        if perceived_input_id in sub_instances:
                            sub_instances[perceived_input_id].append(instance_formula)
                        else:
                            sub_instances[perceived_input_id] = [instance_formula]

        return mapping

    def __extend_node_up_trace_function(self, program_state: ProgramState, parent_trace: ProgramTrace,
                                      focus_node: WildCardNode,
                                      relationship_strategy: RelationshipSearchStrategy) -> Tuple[Node, Dict[int, ProgramTrace]]:

        new_graph = deepcopy(parent_trace.graph)
        new_relationship, _ = relationship_strategy.relationship.copy_and_refresh_id()
        relationship_arg_index = relationship_strategy.known_port
        new_graph.replace_node(focus_node, new_relationship, source_port=PortGraph.MAINTAIN_CONSTANT,
                               target_port=relationship_arg_index, edge_to=True)

        wildcard_nodes_by_rel_port_index: Dict[int, WildCardNode] = {}
        for arg_index, input_type in enumerate(new_relationship.input_types):
            if arg_index != relationship_arg_index:

                # TODO VERY UNSAFE ASSUMPTION - NEED BETTER LANGUAGE SUPPORT FOR ABSTRACT TYPE VALIDITY
                effective_input_type = input_type
                if input_type is Any:
                    effective_input_type = focus_node.value_type

                new_wildcard = WildCardNode(effective_input_type)
                new_graph.add_edge(new_wildcard, new_relationship, to_port=arg_index)
                wildcard_nodes_by_rel_port_index[arg_index] = new_wildcard

        arg_traces = {}
        for port_index, wildcard_node in wildcard_nodes_by_rel_port_index.items():
            arg_trace = ProgramTrace([parent_trace], new_graph, wildcard_node.id, parent_trace.depth + 1, False, None, None,
                                     bool, program_state)
            arg_traces[port_index] = arg_trace

        self._temp_cache__new_relationship = new_relationship
        self._temp_cache__port_to_arg_traces = arg_traces

        return list(arg_traces.values())


class ForkJoinDirective(SynthesisDirective):
    def __init__(self, focus_trace: ProgramTrace, child_trace_1: ProgramTrace, child_trace_2: ProgramTrace):
        super().__init__(DirectiveType.FORK_JOIN)
        self.focus_trace: ProgramTrace = focus_trace
        self.child_trace_1: ProgramTrace = child_trace_1
        self.child_trace_2: ProgramTrace = child_trace_2

    def get_parameterized_trace_ids(self) -> Tuple[int, ...]:
        return tuple(sorted([self.child_trace_1.id, self.child_trace_2.id]))

    def run(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        focus_trace = self.focus_trace
        child_trace_1 = self.child_trace_1
        child_trace_2 = self.child_trace_2
        collected_join_traces = []

        mapping = self.__fork_join_trace_keyed_instances(program_state, evaluation_context, focus_trace, child_trace_1, child_trace_2)

        join_graph_base = self.__fork_join_trace_base_graph_function(focus_trace, child_trace_1, child_trace_2)
        ordered_ids: Tuple[int, int] = (
            child_trace_1.id, child_trace_2.id) if child_trace_1.id < child_trace_2.id else (
            child_trace_2.id, child_trace_1.id)

        for directive_key, new_graph_instances in mapping.items():
            join_trace = program_state.get_trace_graph().get_or_create_traces(
                ordered_ids, directive_key,
                lambda: [self.__fork_join_trace_trace_function(program_state, join_graph_base, focus_trace, child_trace_1,
                                                            child_trace_2, directive_key.duplicate_abstract_nodes)]
            )[0]

            # TODO inconsistency with how get_or_create uses the parent ids internally in the new higl level cache - needs the join id. Recording here while debugging to get unblocked
            program_state.get_trace_graph().record_directive(self.focus_trace.id, BasicSynthesisActions.MERGE_DOWN,
                                                             None, ordered_ids, [join_trace])
            collected_join_traces.append(join_trace)
            join_trace.update_instances(program_state, new_graph_instances, evaluation_context)

        # Handle empty case since there will be no trace graph edge.
        if len(mapping) == 0:
            program_state.get_trace_graph().record_merge_failure(ordered_ids)
            program_state.get_trace_graph().record_directive(self.focus_trace.id, BasicSynthesisActions.MERGE_DOWN, None, ordered_ids, [])

        return collected_join_traces

    def __fork_join_trace_trace_function(self, program_state: ProgramState, base_graph: PortGraph,
                                         parent_trace: ProgramTrace, child_trace_1: ProgramTrace,
                                         child_trace_2: ProgramTrace, duplicate_abstract_nodes: Tuple[Tuple]) -> ProgramTrace:

        join_graph = deepcopy(base_graph)

        for duplicate_pair in duplicate_abstract_nodes:
            replace_node_id = max(duplicate_pair)
            real_node_id = min(duplicate_pair)

            join_graph.replace_node_with_existing_node_all_edges(replace_node_id, real_node_id)

        new_trace = ProgramTrace([child_trace_1, child_trace_2], join_graph, parent_trace.focus_node_id,
                                 max(child_trace_1.depth, child_trace_2.depth), True, parent_trace.latest_fork,
                                 parent_trace.latest_fork_index, parent_trace.root_value_type, program_state)
        # NOTE THIS ONLY APPLIES TO TERMINALS
        # NOTE HOW IN THIS SPECIAL CASE THE FIRST ARG IS THE PARENT! OF THE FOCUS TRACE
        if len(self.focus_trace.parent_ids) > 1:
            raise Exception("Unexpected")
        parent_id = self.focus_trace.parent_ids[0]
        program_state.get_trace_graph().record_concretized_branch(parent_id, new_trace.id)
        return new_trace

    def __fork_join_trace_keyed_instances(self, program_state: ProgramState, evaluation_context: EvaluationContext, focus_trace: ProgramTrace,
                                          child_trace_1: ProgramTrace,
                                          child_trace_2: ProgramTrace) -> TraceInstanceMapping[ForkJoinKey, TraceGraphInstance]:
        mapping: TraceInstanceMapping[ForkJoinKey, TraceGraphInstance] = TraceInstanceMapping()
        for output_id_in_child_1, child_1_graph_instances in child_trace_1.graph_instances(program_state, evaluation_context).items():
            if output_id_in_child_1 in child_trace_2.graph_instances(program_state, evaluation_context):
                child_2_graph_instances = child_trace_2.graph_instances(program_state, evaluation_context)[output_id_in_child_1]

                arg_pair_uuid_to_graph_instances_1: DefaultDict[uuid.UUID, List[TraceGraphInstance]] = defaultdict(list)
                for graph_instance in child_1_graph_instances:
                    uuid_value = graph_instance.fork_indexes[focus_trace.focus_node_id]
                    arg_pair_uuid_to_graph_instances_1[uuid_value].append(graph_instance)

                arg_pair_uuid_to_graph_instances_2: DefaultDict[uuid.UUID, List[TraceGraphInstance]] = defaultdict(list)
                for graph_instance in child_2_graph_instances:
                    uuid_value = graph_instance.fork_indexes[focus_trace.focus_node_id]
                    arg_pair_uuid_to_graph_instances_2[uuid_value].append(graph_instance)

                for uuid_val, graph_instances_1 in arg_pair_uuid_to_graph_instances_1.items():
                    graph_instances_2 = arg_pair_uuid_to_graph_instances_2[uuid_val]

                    for graph_instance_1 in graph_instances_1:
                        for graph_instance_2 in graph_instances_2:

                            duplicate_abstract_nodes_1_2 = set()

                            for concrete_id in graph_instance_1.get_concrete_nodes_included():
                                if graph_instance_2.contains_concrete_node(concrete_id):
                                    abstract_1 = graph_instance_1.get_abstract_from_concrete(concrete_id)
                                    abstract_2 = graph_instance_2.get_abstract_from_concrete(concrete_id)
                                    duplicate_abstract_nodes_1_2.add((abstract_1, abstract_2))

                            duplicate_abstract_nodes_1_2 = tuple(sorted(duplicate_abstract_nodes_1_2))

                            new_graph_instance = TraceGraphInstance.validate_and_merge(graph_instance_1, graph_instance_2,
                                                                                       duplicate_abstract_nodes_1_2)
                            directive_key = ForkJoinKey(duplicate_abstract_nodes_1_2)
                            sub_instances = mapping[directive_key]
                            if output_id_in_child_1 in sub_instances:
                                sub_instances[output_id_in_child_1].append(new_graph_instance)
                            else:
                                sub_instances[output_id_in_child_1] = [new_graph_instance]
        return mapping

    def __fork_join_trace_base_graph_function(self, focus_trace: ProgramTrace,
                                            child_trace_1: ProgramTrace,
                                            child_trace_2: ProgramTrace) -> PortGraph:

        join_graph_base = deepcopy(child_trace_1.graph)

        join_node_edges = child_trace_2.graph.get_edges_to_by_id(focus_trace.focus_node_id)
        index_based_edges = list(filter(lambda e: e.target_port == child_trace_2.latest_fork_index, join_node_edges))
        if len(index_based_edges) != 1:
            raise Exception("Bad edge assumption")
        child_2_wildcard_join_edge: Edge = index_based_edges[0]
        child_2_wildcard_node = join_graph_base.get_node_by_id(child_2_wildcard_join_edge.source_node)

        child_2_concrete_node = child_trace_2.graph.get_node_by_id(child_trace_2.focus_node_id)

        join_graph_base.replace_node(child_2_wildcard_node, child_2_concrete_node,
                                     source_port=child_2_wildcard_join_edge.source_port)

        node_queue = [child_2_concrete_node]
        while node_queue:
            # TODO check for cycles once operators are repeated
            node = node_queue.pop()
            edges_to_node = child_trace_2.graph.get_edges_to_by_id(node.id)
            for edge in edges_to_node:
                source_node = child_trace_2.graph.get_node_by_id(edge.source_node)
                join_graph_base.add_edge(source_node, node, from_port=edge.source_port, to_port=edge.target_port)
                node_queue.append(source_node)

        return join_graph_base


class GenericMergeDirective(SynthesisDirective):
    def __init__(self, traces_to_merge: List[ProgramTrace]):
        super().__init__(DirectiveType.GENERIC_MERGE)
        self.traces_to_merge: List[ProgramTrace] = traces_to_merge

    # TODO this should probably be cached in goals via dfs code
    def get_parameterized_trace_ids(self) -> Tuple[int, ...]:
        return tuple(sorted([trace.id for trace in self.traces_to_merge]))

    def run(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:

        raise NotImplementedError("Need to uncomment the below code AND uncomment dfscodes in trace - THEY ARE TOO SLOW ")

    #     common_dfs_traces = self.traces_to_merge
    #     if len(common_dfs_traces) < 2:
    #         raise Exception("Not meaningful")
    #
    #     if any(common_dfs_traces[0].generic_dfs_code_str != t.generic_dfs_code_str for t in common_dfs_traces):
    #         raise Exception("Bad arg list -  dfs codes")
    #
    #     focus_node_dfs_id = common_dfs_traces[0].generic_dfs_code.get_node_dfs_id(common_dfs_traces[0].focus_node_id)
    #     if any(focus_node_dfs_id != t.generic_dfs_code.get_node_dfs_id(t.focus_node_id) for t in common_dfs_traces):
    #         raise Exception("Bad arg list - focus nodes")
    #
    #     if any((common_dfs_traces[0].depth != t.depth or
    #             common_dfs_traces[0].latest_fork != t.latest_fork or
    #             common_dfs_traces[0].latest_fork_index != t.latest_fork_index or
    #             common_dfs_traces[0].root_value_type != t.root_value_type) for t in common_dfs_traces):
    #         raise Exception("Bad arg list - other")
    #
    #     directive_key = GenericMergeKey(common_dfs_traces[0].generic_dfs_code_str)
    #     trace_ids = tuple(sorted(t.id for t in common_dfs_traces))
    #     result = program_state.get_trace_graph().get_or_create_traces(
    #         trace_ids, directive_key,
    #         lambda: [self.__merge_to_generic_graph_trace_function(program_state, common_dfs_traces)]
    #     )
    #     new_trace, node_lookup = result[0], result[1:]
    #
    #     new_instances: DefaultDict[int, List[TraceGraphInstance]] = defaultdict(list)
    #     for index, trace in enumerate(common_dfs_traces):
    #         for out_id, instances in trace.graph_instances(program_state, evaluation_context).items():
    #             for instance in instances:
    #                 new_instance = instance.copy_and_reassign_abstract_ids(node_lookup[index])
    #                 new_instance.generics_index = index
    #                 new_instances[out_id].append(new_instance)
    #
    #     new_trace.update_instances(program_state, new_instances, evaluation_context)
    #     return [new_trace]
    #
    # def __merge_to_generic_graph_trace_function(self, program_state: ProgramState,
    #                                           common_dfs_traces: List[ProgramTrace]) -> Tuple[ProgramTrace, List[Dict[int, int]]]:
    #
    #     initial_focus_node_id = common_dfs_traces[0].focus_node_id
    #     common_depth = common_dfs_traces[0].depth
    #     common_latest_fork = common_dfs_traces[0].latest_fork
    #     common_latest_fork_index = common_dfs_traces[0].latest_fork_index
    #     common_root_value_type = common_dfs_traces[0].root_value_type
    #
    #     generic_graph, node_lookup = merge_graphs_to_generic_graph(
    #         [(trace.graph, trace.generic_dfs_code) for trace in common_dfs_traces])
    #
    #
    #     new_focus_node_id = node_lookup[0].get(initial_focus_node_id, initial_focus_node_id)
    #
    #     new_trace = ProgramTrace(common_dfs_traces, generic_graph, new_focus_node_id, common_depth, True,
    #                              common_latest_fork, common_latest_fork_index, common_root_value_type, program_state)
    #     new_trace.original_dfs_code_if_generic = common_dfs_traces[0].generic_dfs_code_str
    #
    #     if new_trace.starting_abstract_id in node_lookup[0]:
    #         new_trace.starting_abstract_id = node_lookup[0][new_trace.starting_abstract_id]
    #
    #     key = (new_trace.original_dfs_code_if_generic, new_trace.graph.get_generic_count())
    #     program_state.get_trace_graph().record_generic_trace(common_root_value_type, key, new_trace)
    #     return new_trace, node_lookup


class LeafJoinDirective(SynthesisDirective):
    def __init__(self, trace_1: ProgramTrace, trace_2: ProgramTrace):
        super().__init__(DirectiveType.LEAF_JOIN)
        self.trace_1: ProgramTrace = trace_1
        self.trace_2: ProgramTrace = trace_2

    def get_parameterized_trace_ids(self) -> Tuple[int, ...]:
        return tuple(sorted([self.trace_1.id, self.trace_2.id]))

    def run(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        trace_1 = self.trace_1
        trace_2 = self.trace_2
        collected_join_traces = []

        # TODO validate traces have overlap in debug mode

        if not trace_1.terminal or not trace_2.terminal:
            raise Exception("Traces must be terminal")

        ordered_ids: Tuple[int, int] = (trace_1.id, trace_2.id) if trace_1.id < trace_2.id else (trace_2.id, trace_1.id)
        mapping = self.__leaf_join_traces_keyed_instances(program_state, evaluation_context, trace_1, trace_2)

        base_graph_node_lookup = {trace_1.starting_abstract_id: trace_2.starting_abstract_id}
        base_graph = PortGraph.merge_graphs(trace_1.graph, trace_2.graph, base_graph_node_lookup)[0]

        for duplicate_abstract_nodes, new_graph_instances in mapping.items():
            directive_key = LeafJoinKey(duplicate_abstract_nodes)
            current_trace = program_state.get_trace_graph().get_or_create_traces(
                ordered_ids, directive_key,
                lambda: [self.__leaf_join_trace_function(program_state, base_graph, trace_1, trace_2, duplicate_abstract_nodes)]
            )[0]
            collected_join_traces.append(current_trace)

            current_trace.update_instances(program_state, new_graph_instances, evaluation_context)

        # Handle empty case since there will be no trace graph edge.
        if len(mapping) == 0:
            program_state.get_trace_graph().record_merge_failure(ordered_ids)

        return collected_join_traces

    def __leaf_join_trace_function(self, program_state: ProgramState, base_graph: PortGraph,
                                   child_trace_1: ProgramTrace, child_trace_2: ProgramTrace,
                                   duplicate_abstract_nodes: Tuple[Tuple]) -> ProgramTrace:
        join_graph = deepcopy(base_graph)

        for duplicate_pair in duplicate_abstract_nodes:
            replace_node_id = max(duplicate_pair)
            real_node_id = min(duplicate_pair)

            join_graph.replace_node_with_existing_node_all_edges(replace_node_id, real_node_id)

        return ProgramTrace([child_trace_1, child_trace_2], join_graph, child_trace_1.starting_abstract_id,
                                 max(child_trace_1.depth, child_trace_2.depth), True, None,
                                 None, child_trace_1.root_value_type, program_state)

    def __leaf_join_traces_keyed_instances(self, program_state: ProgramState, evaluation_context: EvaluationContext,
                                           trace_1: ProgramTrace, trace_2: ProgramTrace):
        mapping: TraceInstanceMapping[Tuple[Tuple], TraceGraphInstance] = TraceInstanceMapping()
        for start_id_in_child_1, graph_instances_1 in trace_1.graph_instances(program_state, evaluation_context).items():
            if start_id_in_child_1 in trace_2.graph_instances(program_state, evaluation_context):
                graph_instances_2 = trace_2.graph_instances(program_state, evaluation_context)[start_id_in_child_1]

                for graph_instance_1 in graph_instances_1:
                    for graph_instance_2 in graph_instances_2:

                        duplicate_abstract_nodes_1_2 = set()

                        for concrete_id in graph_instance_1.get_concrete_nodes_included():
                            if concrete_id != start_id_in_child_1:
                                if graph_instance_2.contains_concrete_node(concrete_id):
                                    abstract_1 = graph_instance_1.get_abstract_from_concrete(concrete_id)
                                    abstract_2 = graph_instance_2.get_abstract_from_concrete(concrete_id)
                                    duplicate_abstract_nodes_1_2.add((abstract_1, abstract_2))

                        duplicate_abstract_nodes_1_2 = tuple(sorted(duplicate_abstract_nodes_1_2))

                        new_graph_instance = TraceGraphInstance.validate_and_merge(graph_instance_1, graph_instance_2,
                                                                                   duplicate_abstract_nodes_1_2)
                        sub_instances = mapping[duplicate_abstract_nodes_1_2]
                        if start_id_in_child_1 in sub_instances:
                            sub_instances[start_id_in_child_1].append(new_graph_instance)
                        else:
                            sub_instances[start_id_in_child_1] = [new_graph_instance]

        return mapping


class RefocusDirective(SynthesisDirective):
    def __init__(self, program_trace: ProgramTrace):
        self.program_trace = program_trace
        super().__init__(DirectiveType.REFOCUS_PREDICATE)

    def run(self, program_state: ProgramState, evaluation_context: 'EvaluationContext') -> List[ProgramTrace]:
        # Requirements - program must be terminal, and the current starting abstract id must have more than one outbound edge
        if not self.program_trace.terminal:
            raise Exception("Refocus trace must be terminal")

        staring_node_outbound_edge_count = len(self.program_trace.graph.get_edges_from_by_id(self.program_trace.starting_abstract_id))
        if staring_node_outbound_edge_count < 2:
            raise Exception("Refocus trace current start node must have more than one predicate clause")

        mapping = self.__refocus_keyed_instances(program_state, evaluation_context)
        new_traces = []

        for directive_key, sub_instances in mapping.items():
            new_focus_node_id = directive_key.new_focus_node_id
            new_trace = program_state.get_trace_graph().get_or_create_traces(
                (self.program_trace.id,), directive_key,
                lambda: [self.__refocus_trace_function(program_state, new_focus_node_id)]
            )[0]
            new_traces.append(new_trace)
            new_trace.update_instances(program_state, sub_instances, evaluation_context)

        program_state.get_trace_graph().record_directive(
            self.program_trace.id, BasicSynthesisActions.REFOCUS, None, None, new_traces)
        return new_traces

    def __refocus_keyed_instances(self, program_state: ProgramState,
                                  evaluation_context: EvaluationContext) -> TraceInstanceMapping[RefocusPredicateKey, TraceGraphInstance]:
        mapping: TraceInstanceMapping[RefocusPredicateKey, TraceGraphInstance] = TraceInstanceMapping()
        original_graph_instances = program_state.get_evaluation_matrix().get_trace_instances_by_trace_id(self.program_trace.id)

        for input_node_id in self.program_trace.abstract_input_nodes:
            if input_node_id != self.program_trace.starting_abstract_id:
                directive_key = RefocusPredicateKey(input_node_id)

                # Re-key instances based on new focus node
                for _original_keying_id, sub_instances in original_graph_instances.items():
                    for sub_instance in sub_instances:
                        instance_copy = copy(sub_instance)
                        new_keying_id = instance_copy.get_concrete_from_abstract(input_node_id)

                        if new_keying_id is None:
                            print("WHAT?")

                        if new_keying_id not in mapping[directive_key]:
                            mapping[directive_key][new_keying_id] = [instance_copy]
                        else:
                            mapping[directive_key][new_keying_id].append(instance_copy)

        return mapping

    def __refocus_trace_function(self, program_state: ProgramState,
                                 new_focus_node_id: int) -> ProgramTrace:
        graph_copy = deepcopy(self.program_trace.graph)
        new_trace = ProgramTrace([self.program_trace], graph_copy,
                                focus_node_id=new_focus_node_id,
                                depth=self.program_trace.depth,
                                terminal=True,
                                latest_fork=None,
                                latest_fork_index=None,
                                root_value_type=self.program_trace.root_value_type,
                                program_state=program_state,
                                search_direction_forward=self.program_trace.search_direction_forward,
                                predicate_mode=self.program_trace.predicate_mode)
        new_trace.starting_abstract_id = new_focus_node_id
        new_trace.refocused = True
        return new_trace

    def get_parameterized_trace_ids(self) -> Tuple[int, ...]:
        return (self.program_trace.id,)


### WIP - Anonymizing directives, make this about known dsl nodes and Extension (up/down) Merging (up/down)


EXTENSION_DOWN_DIRECTIVES: List[ExtensionSynthesisDirective] = [
    ExtendDownConstantDirective,
    ExtendDownInputDirective,
    ExtendDownConstructorDirective,
    ExtendDownOperatorDirective,
]


EXTENSION_DOWN_TEMPLATES: Dict[Tuple[Type[SynthesisDirective],Type[Node]], Callable[[ProgramTrace], ExtensionSynthesisDirective]] = {
    (ExtendDownConstantDirective, Constant): ExtendDownConstantDirective,
    (ExtendDownInputDirective, InputNode): ExtendDownInputDirective,
    (ExtendDownConstructorDirective, ConstructorOperator): ExtendDownConstructorDirective,
}


for operator_node in OPERATOR_TO_STRATEGY.keys():

    def inline(tt, cls_arg=operator_node.__class__):
        return ExtendDownOperatorDirective(tt, cls_arg)

    EXTENSION_DOWN_TEMPLATES[(ExtendDownOperatorDirective, operator_node.__class__)] = inline


EXTENSION_UP_DIRECTIVES: List[ExtensionSynthesisDirective] = [
    ExtendUpEdgeDirective,
    ExtendUpNodeDirective
]

MERGE_DOWN_DIRECTIVES: List[ExtensionSynthesisDirective] = [
    ForkJoinDirective
]

MERGE_UP_DIRECTIVES: List[ExtensionSynthesisDirective] = [
    ForkJoinDirective,
]
