from collections import defaultdict
from numbers import Number
from random import random
from typing import Type, Tuple, List, Set, Dict, DefaultDict, Callable, Any

from nodes import InputNode, OutputNode
from port_graphs import PortGraph
from interpreter.interpreter import Interpreter
from interpreter.interpreter import GraphIO
from synthesis.design_time_models import WildCardNode
from synthesis.program_state import PerceptionModel
from synthesis.program_state import ProgramState, EvaluationContext
from synthesis.program_state import TraceEvaluation, ProgramTrace, TraceGraphInstance
from synthesis.synthesis_engine import SynthesisDirective, LeafJoinDirective, GenericMergeDirective, ForkJoinDirective, \
    ExtensionSynthesisDirective
from synthesis.synthesis_models import DirectiveType, BasicSynthesisActions


# Simpler alternative to solo concrete concepts
class AbstractNodeSoloInstances:

    def __init__(self, program_state: ProgramState, evaluation_context: EvaluationContext,
                 trace: ProgramTrace, abstract_node_id: int):

        trace_evaluation: TraceEvaluation = program_state.get_evaluation_matrix().get_evaluation(trace.id, evaluation_context)

        if not trace_evaluation.full:
            raise Exception("Trace must be fully evaluated to create solo instances")

        # TODO use this to invalidate later on if perception changes
        self.__perception_in_tracker = evaluation_context.perception_in_id
        self.__perception_out_tracker = evaluation_context.perception_out_id
        self.__perceived_value_ids_with_redundancies: List[Set[int]] = [
            set(inst.get_concrete_from_abstract(abstract_node_id)
                for inst in redundant_instances) for redundant_instances
            in trace_evaluation.get_instances(program_state, evaluation_context.perceived_subset).values()]
        self.__no_redundancies = all(len(s) == 1 for s in self.__perceived_value_ids_with_redundancies)
        self.__flattened_perceived_value_ids: Set[int] = set(i
                                                             for vs in self.__perceived_value_ids_with_redundancies
                                                             for i in vs)

    def temp_get_flat_ids(self):
        return self.__flattened_perceived_value_ids

    def get_miss_overshoot(self, other: 'AbstractNodeSoloInstances') -> Tuple[Set[int], Set[int]]:
        if self.__no_redundancies and other.__no_redundancies:
            miss = self.__flattened_perceived_value_ids.difference(other.__flattened_perceived_value_ids)
            overshoot = other.__flattened_perceived_value_ids.difference(self.__flattened_perceived_value_ids)
            return miss, overshoot

        # Case self has redundancies other does not
        elif not self.__no_redundancies and other.__no_redundancies:
            miss = set()
            for redundant_id_set in self.__perceived_value_ids_with_redundancies:
                if not any(redundant_id in other.__flattened_perceived_value_ids for redundant_id in redundant_id_set):
                    miss.update(redundant_id_set)

            overshoot = other.__flattened_perceived_value_ids.difference(self.__flattened_perceived_value_ids)
            return miss, overshoot

        # Case self has no redundancies other does though
        elif self.__no_redundancies and not other.__no_redundancies:
            miss = self.__flattened_perceived_value_ids.difference(other.__flattened_perceived_value_ids)
            overshoot = set()
            # for redundant_id_set in other.__perceived_value_ids_with_redundancies:
            #     if not any(redundant_id in self.__flattened_perceived_value_ids for redundant_id in redundant_id_set):
            #         overshoot.update(redundant_id_set)
            # TODO maybe this is what it should be?? Makes some pass, but more fail - In think the order cannot be guaranteed...
            for redundant_id_set in other.__perceived_value_ids_with_redundancies:
                mini_overshoot = redundant_id_set.difference(self.__flattened_perceived_value_ids)
                overshoot.update(mini_overshoot)
            return miss, overshoot

        # Case both have redundancies
        # NOTE same impl as the case where self has redundancies and other does not

        # MISS check
        # Compare local self set to global other set - if no matches, add full local set to miss
        # If matches - find the other's local sets which match

        # Overshoot - just take the general overshoot - its not critical yet...
        miss = set()
        for redundant_id_set in self.__perceived_value_ids_with_redundancies:
            if not any(redundant_id in other.__flattened_perceived_value_ids for redundant_id in redundant_id_set):
                miss.update(redundant_id_set)

        overshoot = other.__flattened_perceived_value_ids.difference(self.__flattened_perceived_value_ids)
        return miss, overshoot


class HypothesisPredicateLane:

    def __init__(self, program_state: ProgramState, evaluation_context: EvaluationContext,
                 trace: ProgramTrace,
                 abstract_node_id: int):

        if abstract_node_id not in trace.abstract_input_nodes:
            raise Exception("Abstract node id must be one of the trace's abstract")

        self.trace_id = trace.id
        self.abstract_node_id = abstract_node_id
        self.abstract_solo_instances = AbstractNodeSoloInstances(program_state, evaluation_context, trace, abstract_node_id)

        # Elements are tuples (predicate trace id, abstract node id in predicate)
        self.proposed_predicates: List[Tuple[int, int]] = []
        self.confirmed_predicates: List[Tuple[int, int]] = []
        self.failed_predicates: List[Tuple[int, int]] = []

    def get_key(self) -> Tuple[int, int]:
        return self.trace_id, self.abstract_node_id


class ProgramHypothesis:

    def __init__(self, perception_model_in: PerceptionModel, perception_model_out: PerceptionModel):
        self.perception_model_in: PerceptionModel = perception_model_in
        self.perception_model_out: PerceptionModel = perception_model_out

        # Trace and concept lookup by id
        self.__proposed_operator_traces_by_type: DefaultDict[Type, Dict[int, ProgramTrace]] = defaultdict(dict)
        self.__perceived_output_ids_to_proposed_op_by_type: DefaultDict[Type, Dict[int, int]] = defaultdict(dict)
        self.confirmed_operator_traces: DefaultDict[Type, Dict[int, ProgramTrace]] = defaultdict(dict)
        self.predicate_traces: Dict[int, ProgramTrace] = {}

        # Values are Tuple of (op trace id, predict trace id, abstract node id in predicate)
        self.committed_op_and_predicate_traces: Set[Tuple[int, int, int]] = set()

        # Track predicate reuse across lanes: predicate_trace_id -> List of (lane_key, abstract_node_id)
        self.predicate_multi_lane_usage: DefaultDict[int, List[Tuple[Tuple[int, int], int]]] = defaultdict(list)

        # self.solo_concepts: Dict[int, ConcreteConceptSolo] = {}

        # Key is Tuple of (operator trace id, abstract node id in operator)
        self.op_nodes_to_predicate_lanes: Dict[Tuple[int, int], HypothesisPredicateLane] = {}

        self.temp_predicate_success = False

        # Value is Tuple of (operator trace id, abstract node id in operator)
        # self.solo_concepts_to_operators: Dict[int, Tuple[int, int]] = {}

        # Value is list of tuples of (predicate trace id, abstract node id in predicate)
        # self.solo_concepts_to_predicate_lanes: DefaultDict[int, List[Tuple[int, int]]] = {}

        # self.output_concepts: List[Any] = []
        # self.input_concepts: List[Any] = []
        # Need some kind of mapping of operator traces to predicates

        # TODO build out resources to link + measure op traces with predicates based on common instances

    def propose_op_trace(self, program_state: ProgramState, evaluation_context: EvaluationContext,
                         op_trace: ProgramTrace):

        # TODO the behavior in this method should be a part of a flexible policy - hypothesis action space

        trace_eval: TraceEvaluation = program_state.get_evaluation_matrix().get_evaluation(op_trace.id, evaluation_context)
        if not trace_eval.full:
            raise Exception("Cannot propose an operator trace which is not fully evaluated")

        value_type = op_trace.root_value_type
        output_ids_to_proposed_op_id: Dict[int, int] = self.__perceived_output_ids_to_proposed_op_by_type[value_type]
        trace_output_ids = trace_eval.instance_ids

        collisions_to_out_ids: DefaultDict[int, Set[int]] = defaultdict(set)
        for out_id in trace_output_ids:
            colliding_trace_id = output_ids_to_proposed_op_id.get(out_id)
            if colliding_trace_id is not None:
                collisions_to_out_ids[colliding_trace_id].add(out_id)


        beat_all_collisions = all((program_state.get_evaluation_matrix().get_evaluation(op_trace.id, evaluation_context).eval_posterior >
                                   program_state.get_evaluation_matrix().get_evaluation(colliding_trace_id, evaluation_context).eval_posterior)
                                  for colliding_trace_id in collisions_to_out_ids.keys())
        if len(collisions_to_out_ids) == 0:
            self.__proposed_operator_traces_by_type[value_type][op_trace.id] = op_trace
            for out_id in trace_output_ids:
                output_ids_to_proposed_op_id[out_id] = op_trace.id

        # Only if you beat all the collisions - replace them all...
        elif beat_all_collisions:

            # Replace all the collisions
            for colliding_trace_id, out_ids in collisions_to_out_ids.items():
                for out_id in out_ids:
                    output_ids_to_proposed_op_id[out_id] = op_trace.id

            # Remove all non-collisions from the previous trace
            for out_ids, op_id in list(output_ids_to_proposed_op_id.items()):
                if op_id in collisions_to_out_ids.keys():
                    for out_id in out_ids:
                        del output_ids_to_proposed_op_id[out_id]

            for op_id in collisions_to_out_ids.keys():
                del self.__proposed_operator_traces_by_type[value_type][op_id]

            # Add the new one
            self.__proposed_operator_traces_by_type[value_type][op_trace.id] = op_trace

    def has_op_coverage(self, program_state: ProgramState, value_type: Type) -> bool:
        # NOTE need to keep the filter because the method called is not implemented well
        output_value_count = len(list(filter(lambda v: v.output, program_state.get_perception_xref().get_all_perceived_values(self.perception_model_out.id))))
        if output_value_count == len(self.__perceived_output_ids_to_proposed_op_by_type[value_type]):
            return True
        return False

    def submit_op_traces_new(self, program_state: ProgramState, evaluation_context: EvaluationContext, value_type: Type):
        self.submit_op_traces(program_state, evaluation_context, list(self.__proposed_operator_traces_by_type[value_type].values()))

    def submit_op_traces(self, program_state: ProgramState, evaluation_context: EvaluationContext, op_traces: List[ProgramTrace]):
        for t in op_traces:
            self.confirmed_operator_traces[t.root_value_type][t.id] = t
            for abstract_node_id in t.abstract_input_nodes:
                lane = HypothesisPredicateLane(program_state, evaluation_context, t, abstract_node_id)
                self.op_nodes_to_predicate_lanes[lane.get_key()] = lane

    def create_master_graphs_from_hypothesis(self, output_type: Type) -> List[PortGraph]:

        # TODO validate more than this...
        if not self.temp_predicate_success:
            raise Exception("Cannot create master graphs from a hypothesis without predicate success")

        master_graphs = []
        # For each op trace
        for op_trace_id, op_trace in self.confirmed_operator_traces[output_type].items():

            lane_tuple_list = list(filter(lambda t: t[0][0] == op_trace_id, self.op_nodes_to_predicate_lanes.items()))
            op_nodes_to_predicates: Dict[int, Tuple[int, int]] = {lane_tuple[0][1]: lane_tuple[1].confirmed_predicates[0]
                                                                  for lane_tuple in lane_tuple_list}

            # Ensure_all predicate/node combos are unique
            if len(op_nodes_to_predicates) != len(set(op_nodes_to_predicates.values())):
                raise Exception("Cannot create master graphs from a hypothesis with non-unique predicate/node combos")

            # Group by predicate trace id
            predicate_id_to_node_lookup: DefaultDict[int, Dict[int, int]] = defaultdict(dict)
            for op_node_id, (predicate_id, predicate_node_id) in op_nodes_to_predicates.items():
                predicate_id_to_node_lookup[predicate_id][predicate_node_id] = op_node_id

            combo_graph = op_trace.graph
            for predicate_trace_id, node_lookup in predicate_id_to_node_lookup.items():
                predicate_graph = self.predicate_traces[predicate_trace_id].graph
                combo_graph = PortGraph.merge_graphs(predicate_graph, combo_graph, node_lookup)[0]

            # for lane_key, lane in :
            #     operator_node_id = lane_key[1]
            #     predicate_id, predicate_node_id = lane.confirmed_predicates[0]
            #     node_lookup = {predicate_node_id: operator_node_id}
            #     predicate_graph = self.predicate_traces[predicate_id].graph
            #     combo_graph = DslGraph.merge_graphs(predicate_graph, combo_graph, node_lookup)[0]

            master_graphs.append(combo_graph)

        return master_graphs


def bootstrap_trace(program_state: ProgramState,
                    evaluation_context: EvaluationContext,
                    value_type: Type,
                    predicate_mode: bool) -> ProgramTrace:

    output_side = not predicate_mode

    root_graph = PortGraph()
    if output_side:
        root_node = OutputNode(value_type=value_type)
    else:
        root_node = InputNode(value_type=value_type, perception_qualifiers=None)
    root_wildcard = WildCardNode(value_type)

    if predicate_mode:
        root_graph.add_node(root_node)
    else:
        root_graph.add_edge(root_wildcard, root_node)

    root_id = root_node.id if predicate_mode else root_wildcard.id

    graph_instances: Dict[int, List[TraceGraphInstance]] = {}
    perception_key = evaluation_context.perception_out_id if output_side else evaluation_context.perception_in_id
    for perceived_value in filter(lambda p_v: p_v.output == output_side and p_v.value_type == value_type,
                                  program_state.get_perception_xref().get_all_perceived_values(perception_key)):

        state: Dict[int, Any] = {root_node.id: perceived_value.value}
        input_abstract_to_concrete: Dict[int, int] = None

        if predicate_mode:
            input_abstract_to_concrete = {root_node.id: perceived_value.id}
        else:
            state[root_wildcard.id] = perceived_value.value

        trace = TraceGraphInstance(root_id, perceived_value.id, {}, state,
                                   abstract_to_concrete_starter=input_abstract_to_concrete)
        graph_instances[perceived_value.id] = [trace]

    output_value_type = bool if predicate_mode else value_type
    root_trace = ProgramTrace([], root_graph, root_id, 0 if predicate_mode else 1, False, None,
                              None, output_value_type, program_state,
                              search_direction_forward=predicate_mode,
                              predicate_mode=predicate_mode)
    program_state.get_trace_graph().add_root_trace(root_trace)
    root_trace.update_instances(program_state, graph_instances, evaluation_context)
    return root_trace


# Returns tuple of directive and focus trace
def predicate_directive_selection(program_state: ProgramState,
                                  evaluation_context: EvaluationContext,
                                  draft_program: ProgramHypothesis) -> List[SynthesisDirective]:
    trace_graph = program_state.get_trace_graph()

    # Get all unconfirmed lanes, not just the first one
    unconfirmed_lanes = sorted(filter(lambda o: len(o.confirmed_predicates) == 0,
                                     draft_program.op_nodes_to_predicate_lanes.values()),
                              key=lambda o: o.get_key())

    if not unconfirmed_lanes:
        return []

    focus_lane = unconfirmed_lanes[0]
    focus_abstract_solo_node_instances: AbstractNodeSoloInstances = focus_lane.abstract_solo_instances

    refocus_traces = list(filter(lambda t: t.terminal and t.
                                 root_value_type == bool and
                                 len(t.abstract_input_nodes) > 1 and
                                 len(t.graph.get_edges_from_by_id(t.starting_abstract_id)) > 1 and
                                 program_state.get_evaluation_matrix().get_evaluation(t.id, evaluation_context).eval_posterior > 0
                                 ,trace_graph.get_all_traces()))
    for refocus_trace in refocus_traces:
        if not program_state.get_trace_graph().synthesis_action_already_run(refocus_trace.id, BasicSynthesisActions.REFOCUS, None, None):

            if refocus_trace.refocused:
                continue

    total_predicate_count = len(list(filter(lambda t: t.root_value_type == bool,program_state.get_trace_graph().get_all_traces())))
    proposal_proportion = len(focus_lane.proposed_predicates) / total_predicate_count if total_predicate_count > 0 else 0
    if random() < proposal_proportion:

        # Score proposed predicates by multi-lane coverage
        proposed_scores = []
        for trace_id, abstract_id in focus_lane.proposed_predicates:
            trace = trace_graph.get_trace(trace_id)
            if trace.starting_abstract_id != abstract_id:
                continue

            # Count how many lanes this proposed predicate can cover
            lane_coverage = 0
            for lane in unconfirmed_lanes:
                key = (trace_id, abstract_id)
                if key in lane.proposed_predicates or key in lane.confirmed_predicates:
                    lane_coverage += 1

            overshoot_len = len(focus_abstract_solo_node_instances.get_miss_overshoot(
                AbstractNodeSoloInstances(program_state, evaluation_context, trace, abstract_id))[1])
            posterior = program_state.get_evaluation_matrix().get_evaluation(trace_id, evaluation_context).eval_posterior

            # Score by: lane_coverage (primary), overshoot (inverse, secondary), posterior (tertiary)
            proposed_scores.append((trace_id, abstract_id, lane_coverage, -overshoot_len, posterior))

        # Sort by multi-lane coverage first
        proposed_scores.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)

        for trace_id, abstract_id, _, _, _ in proposed_scores:
            for other_trace_id, other_abstract_id, _, _, _ in proposed_scores:
                if trace_id >= other_trace_id:
                    continue

                key = (trace_id, other_trace_id) if trace_id < other_trace_id else (other_trace_id, trace_id)
                if program_state.get_trace_graph().check_existing_merge(key, DirectiveType.LEAF_JOIN):
                    continue

                return [LeafJoinDirective(trace_graph.get_trace(trace_id),
                                          trace_graph.get_trace(other_trace_id))]


    # Score extension candidates by multi-lane coverage
    extension_candidates = []
    for trace in filter(lambda t: t.can_extend() and t.root_value_type == bool,
                       program_state.get_trace_graph().get_all_traces()):
        if trace.starting_abstract_id is None:
            continue

        # Count how many lanes this trace can satisfy
        lane_coverage_count = 0
        exact_match_count = 0
        for lane in unconfirmed_lanes:
            trace_instances = AbstractNodeSoloInstances(program_state, evaluation_context,
                                                       trace, trace.starting_abstract_id)
            miss, overshoot = lane.abstract_solo_instances.get_miss_overshoot(trace_instances)
            if len(miss) == 0:
                lane_coverage_count += 1
                if len(overshoot) == 0:
                    exact_match_count += 1

        if lane_coverage_count > 0:
            posterior = program_state.get_evaluation_matrix().get_evaluation(trace.id, evaluation_context).eval_posterior
            # Score by: lane_coverage_count (primary), exact_match_count (secondary), posterior (tertiary)
            extension_candidates.append((trace, lane_coverage_count, exact_match_count, posterior))

    # Sort by multi-lane coverage first, then exact matches, then posterior
    extension_candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

    if extension_candidates:
        best_trace = extension_candidates[0][0]
        return best_trace.get_remaining_extension_directives()

    return []


def predicate_complete_condition(program_state: ProgramState, evaluation_context: EvaluationContext,
                                 draft_program: ProgramHypothesis, previous_directive_results: List[ProgramTrace]) -> bool:

    complete_programs = sorted(filter(lambda t: (t.root_value_type == bool and
                                                 t.wildcard_count == 0 and
                                                 program_state.get_evaluation_matrix().get_evaluation(t.id, evaluation_context).eval_posterior > 0 and
                                                 t.terminal and
                                                 program_state.get_evaluation_matrix().get_evaluation(t.id, evaluation_context).eval_case_count == program_state.get_total_case_count() and
                                                 program_state.get_evaluation_matrix().get_evaluation(t.id, evaluation_context).full),
                                      previous_directive_results),
                               key=lambda t: program_state.get_evaluation_matrix().get_evaluation(t.id, evaluation_context).eval_posterior, reverse=True)

    # Score predicates by how many lanes they can cover
    predicate_lane_coverage: DefaultDict[int, List[Tuple[HypothesisPredicateLane, int, bool]]] = defaultdict(list)
    unconfirmed_lanes = sorted(filter(lambda l: len(l.confirmed_predicates) == 0,
                                     draft_program.op_nodes_to_predicate_lanes.values()),
                               key=lambda o: o.get_key())

    # First pass: evaluate all predicate-lane combinations
    for predicate_program in complete_programs:
        for predicate_abstract_id in predicate_program.abstract_input_nodes:
            key = (predicate_program.id, predicate_abstract_id)

            for lane in unconfirmed_lanes:
                trace_specific_key = (lane.trace_id, *key)
                if trace_specific_key in draft_program.committed_op_and_predicate_traces:
                    continue

                predicate_solo_instance = AbstractNodeSoloInstances(program_state, evaluation_context,
                                                                   predicate_program, predicate_abstract_id)
                miss, overshoot = lane.abstract_solo_instances.get_miss_overshoot(predicate_solo_instance)

                if len(miss) == 0:
                    # Store coverage info: lane, abstract_id, is_exact_match
                    is_exact = len(overshoot) == 0
                    predicate_lane_coverage[predicate_program.id].append((lane, predicate_abstract_id, is_exact))

    # Sort predicates by coverage count (prefer predicates that cover more lanes)
    sorted_predicates = sorted(predicate_lane_coverage.items(),
                               key=lambda x: (len(x[1]), sum(1 for _, _, exact in x[1] if exact)),
                               reverse=True)

    # Second pass: apply predicates prioritizing multi-lane coverage
    for predicate_id, lane_coverage_list in sorted_predicates:
        predicate_program = next(p for p in complete_programs if p.id == predicate_id)
        draft_program.predicate_traces[predicate_program.id] = predicate_program

        # Group by abstract ID to maximize reuse of same predicate with different nodes
        by_abstract_id: DefaultDict[int, List[Tuple[HypothesisPredicateLane, bool]]] = defaultdict(list)
        for lane, abstract_id, is_exact in lane_coverage_list:
            by_abstract_id[abstract_id].append((lane, is_exact))

        # Process each abstract ID, preferring those that cover more lanes
        for abstract_id, lanes_and_exact in sorted(by_abstract_id.items(),
                                                   key=lambda x: len(x[1]), reverse=True):
            key = (predicate_program.id, abstract_id)

            for lane, is_exact in lanes_and_exact:
                # Skip if lane already has a confirmed predicate
                if len(lane.confirmed_predicates) > 0:
                    continue

                trace_specific_key = (lane.trace_id, *key)
                if trace_specific_key in draft_program.committed_op_and_predicate_traces:
                    continue

                if is_exact:
                    if key not in lane.confirmed_predicates:
                        # Check if the predicate holds against the test case
                        if predicate_has_matches_in_test(program_state, evaluation_context,
                                                         predicate_program, abstract_id):
                            lane.confirmed_predicates.append(key)
                            draft_program.committed_op_and_predicate_traces.add(trace_specific_key)
                            # Track multi-lane usage
                            draft_program.predicate_multi_lane_usage[predicate_program.id].append((lane.get_key(), abstract_id))
                        else:
                            lane.failed_predicates.append(key)
                            if key in lane.proposed_predicates:
                                lane.proposed_predicates.remove(key)
                else:
                    if key not in lane.proposed_predicates:
                        lane.proposed_predicates.append(key)

                        # print("ADDING CONFIRMED PREDICATE", lane, predicate_program, predicate_abstract_id)

    if all(len(lane.confirmed_predicates) > 0 for lane in draft_program.op_nodes_to_predicate_lanes.values()):
        draft_program.temp_predicate_success = True
        return True

    return False


def predicate_has_matches_in_test(program_state: ProgramState, evaluation_context: EvaluationContext,
                                  combined_trace: ProgramTrace,
                                  focus_id: int) -> bool:
    # Run the conditions against the test input perceived objects

    new_graph = PortGraph()

    main_input_node = None
    abstract_node_mapping = {}
    perception_set = set(program_state.get_perception_xref()
                         .get_perception_model(evaluation_context.perception_in_id).perception_functions)
    for abstract_input_id in combined_trace.abstract_input_nodes:
        existing_input_node: InputNode = combined_trace.graph.get_node_by_id(abstract_input_id)
        input_node = InputNode(existing_input_node.value_type, perception_set)
        new_graph.add_node(input_node)
        abstract_node_mapping[abstract_input_id] = input_node.id
        if abstract_input_id == focus_id:
            main_input_node = input_node

    if main_input_node is None:
        raise Exception("Focus ID not found in concept trace abstract object IDs.")

    # Add direct edge from input to output, so the graph returns all inputs (without operation) that meet criteria
    output_node = OutputNode(main_input_node.value_type)
    new_graph.add_edge(main_input_node, output_node)

    prepared_graph, _ = PortGraph.merge_graphs(combined_trace.graph, new_graph, abstract_node_mapping)

    # TODO check all cases
    graph_state = GraphIO(prepared_graph)
    all_objects = [perceived_val.value for perceived_val in
                   program_state.get_perception_xref().get_all_perceived_test_values(evaluation_context.perception_in_id)
                   if perceived_val.value_type == main_input_node.value_type]
    graph_state = Interpreter.populate_graph_io_inputs(prepared_graph, graph_state, all_objects)
    try:
        interpreter = Interpreter(initial_depth=1)
        out_state = interpreter.evaluate_port_graph(prepared_graph, graph_state)
        if len(out_state.get_output_values()[output_node.id]) > 0:
            return True
    except Exception as e:
        print("test trial exception caught", e)

    return False


def check_extension_condition(program_subset: List[ProgramTrace],
                              root_value_type: Type,
                              trace_predicate: Callable[[ProgramTrace], bool],
                              trace_sort: Callable[[ProgramTrace], Number],
                              accumulate: bool = False) -> List[ExtensionSynthesisDirective]:
    all_directives = []
    for program in sorted(filter(lambda trace: trace.root_value_type == root_value_type and
                                               trace.can_extend() and
                                               trace_predicate(trace),
                                 program_subset), key=trace_sort, reverse=True):
            new_directives = program.get_remaining_extension_directives()
            if accumulate:
                all_directives.extend(new_directives)
            else:
                return new_directives

    return all_directives


def check_fork_join_condition(program_state: ProgramState,
                              program_subset: List[ProgramTrace],
                              root_value_type: Type,
                              trace_predicate: Callable[[ProgramTrace], bool],
                              trace_sort: Callable[[ProgramTrace], Number],
                              accumulate: bool = False) -> List[ForkJoinDirective]:

    all_directives = []

    # Search for joins
    # Find a trace where the focus is not a wildcard
    # Find all traces where the last fork id is the focus id
    # Return the first pair that: are not the same trace, are not the same fork index, and have not been tried via history
    for join_trace in sorted(filter(lambda trace: trace.root_value_type == root_value_type and
                                           trace.can_merge_as_join_focus() and
                                           trace_predicate(trace),
                             program_subset), key=trace_sort, reverse=True):

        # TODO this sort can be smarter; more "planning"
        potential_child_traces = list(sorted(filter(lambda trace: trace.root_value_type == root_value_type and
                                                                  trace.can_merge() and
                                                                  trace.latest_fork == join_trace.id
                                                                  and trace_predicate(trace),
                                                    program_subset),
                                             key=trace_sort, reverse=True))
        for potential_child_1 in potential_child_traces:

            # TODO JOIN GENERICS
            if potential_child_1.graph.has_generics():
                continue

            for potential_child_2 in potential_child_traces:

                if potential_child_1.id == potential_child_2.id:
                    continue

                # TODO JOIN GENERICS
                if potential_child_2.graph.has_generics():
                    continue

                if potential_child_1.latest_fork_index != potential_child_2.latest_fork_index:
                    history_search_tuple = (potential_child_1.id, potential_child_2.id) if potential_child_1.id < potential_child_2.id else (potential_child_2.id, potential_child_1.id)
                    if not program_state.get_trace_graph().check_existing_merge(history_search_tuple, DirectiveType.FORK_JOIN):
                        directive = ForkJoinDirective(join_trace, potential_child_1, potential_child_2)
                        if accumulate:
                            all_directives.append(directive)
                        else:
                            return [directive]
    return all_directives


def default_trace_predicate(program_state: ProgramState,
                            evaluation_context: EvaluationContext,
                            program_trace: ProgramTrace,
                            min_case_count: int) -> bool:
    return (min_case_count <= program_state.get_evaluation_matrix()
            .get_evaluation(program_trace.id, evaluation_context).eval_case_count)


def default_trace_sort(program_state: ProgramState,
                           evaluation_context: EvaluationContext,
                           program_trace: ProgramTrace) -> Number:
    return program_state.get_evaluation_matrix().get_evaluation(program_trace.id, evaluation_context).eval_posterior


def check_generic_merge_condition(program_state: ProgramState, evaluation_context: EvaluationContext,
                                  root_value_type: Type) -> List[GenericMergeDirective]:
    # TODO this method has been significantly refactored without tesing. 99% chance it does not work. Do not be fooled.
    if random() < 0.0:
        group_by_dfs_code = defaultdict(list)
        for trace in program_state.get_trace_graph().get_all_traces():
            # NOTE disallowing overlapping generics for now
            if trace.root_value_type == root_value_type and not trace.graph.has_generics() and trace.terminal: # TODO not sure this needs to be terminal
                group_by_dfs_code[(trace.generic_dfs_code_str, trace.latest_fork, trace.latest_fork_index)].append(trace)

        group_by_dfs_code_with_metadata = defaultdict(list)
        for dfs_code_and_fork, group in group_by_dfs_code.items():
            dfs_code = dfs_code_and_fork[0]
            key = (dfs_code, len(group))
            if program_state.get_trace_graph().get_generic_traces(key):
                continue

            total_covered = set()
            coverage_by_case = defaultdict(list)
            for trace in group:
                for out_id in program_state.get_evaluation_matrix().get_trace_instances_by_trace_id(trace.id).keys():
                    total_covered.add(out_id)
                    case = program_state.get_perception_xref().get_perceived_value(out_id).case_index
                    coverage_by_case[case].append(trace.id)

            cumulative_coverage = 1
            for case in range(program_state.get_total_case_count()):
                coverage = coverage_by_case[case]

                if len(coverage) == 0:
                    cumulative_coverage = 0
                    break

                coverage_ratio =  len(set(coverage)) / len(coverage)
                if coverage_ratio > 1:
                    coverage_ratio = 0 # 1 / coverage_ratio # Penalize duplicates TODO temp experiment...

                cumulative_coverage *= coverage_ratio

            coverage_efficiency = len(total_covered)/len(group)
            group_by_dfs_code_with_metadata[dfs_code] = (cumulative_coverage, coverage_efficiency, group)

        worthwhile_generics = list(filter(lambda m: m[0] == 1.0 and m[1] > 1 and len(m[2]) > 1, group_by_dfs_code_with_metadata.values()))

        if len(worthwhile_generics) > 0:
            # TODO sort by posterior before picking the first
            for worthwhile_generic in worthwhile_generics:
                return [GenericMergeDirective(worthwhile_generic[2])]

    return []
    ### END GENERIC CHECK ###


def is_complete_fit_op_program(program_state: ProgramState, evaluation_context: EvaluationContext,
                               trace: ProgramTrace,
                               require_full_eval = True) -> List[ProgramTrace]:
    return (trace.terminal and
            trace.wildcard_count == 0 and
            program_state.get_evaluation_matrix().get_evaluation(trace.id, evaluation_context).eval_case_count == program_state.get_total_case_count() and
            program_state.get_evaluation_matrix().get_evaluation(trace.id, evaluation_context).eval_posterior > 0 and
            (program_state.get_evaluation_matrix().get_evaluation(trace.id, evaluation_context).full or not require_full_eval))


def operator_complete_condition(program_state: ProgramState,
                                evaluation_context: EvaluationContext,
                                draft_program: ProgramHypothesis,
                                root_value_type: Type) -> bool:

    # Update complete graphs dict
    node_coverage: Set[int] = set()
    complete_fit_programs: List[ProgramTrace] = list(filter(lambda t: t.root_value_type == root_value_type and is_complete_fit_op_program(program_state, evaluation_context, t),
                                                            program_state.get_trace_graph().get_all_traces()))

    output_node_count = len(list(filter(lambda perceived_val: perceived_val.output and perceived_val.value_type == root_value_type,
        program_state.get_perception_xref().get_all_perceived_values(evaluation_context.perception_out_id))))

    for program_trace in complete_fit_programs:
        # context.data.complete_programs[program_trace.id] = program_trace
        for node_covered in (program_state.get_evaluation_matrix().get_evaluation(program_trace.id, evaluation_context)
                .get_instances(program_state, evaluation_context.perceived_subset).keys()):
            node_coverage.add(node_covered)

    if len(node_coverage) == output_node_count:

        # See if you can pick a valid combination of programs
        coverage = set()
        disjoint_sub_programs = []
        complete_programs_sorted_posterior = sorted(complete_fit_programs, key=lambda p: program_state.get_evaluation_matrix().get_evaluation(p.id, evaluation_context).eval_posterior,
                                                    reverse=True)
        for program in complete_programs_sorted_posterior:
            graph_instance_keys = (program_state.get_evaluation_matrix().get_evaluation(program.id, evaluation_context)
                .get_instances(program_state, evaluation_context.perceived_subset).keys())
            if all(out_id not in coverage for out_id in graph_instance_keys):
                disjoint_sub_programs.append(program)
                for out_id in graph_instance_keys:
                    coverage.add(out_id)
            if len(coverage) == output_node_count:
                break
        if len(coverage) == output_node_count:
            # NOTE MUST REASSIGN, NOT EXTEND BECAUSE THE CONTEXT IS REUSED FOR ENV SHAPES.
            # context.data.disjoint_sub_programs = disjoint_sub_programs
            for program in disjoint_sub_programs:
                draft_program.propose_op_trace(program_state, evaluation_context, program)
            draft_program.submit_op_traces_new(program_state, evaluation_context, root_value_type)
            return True

    return False
