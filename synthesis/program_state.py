from collections import defaultdict
from copy import copy
from typing import Dict, Set, DefaultDict, Tuple, Callable, List, Optional, Type, Union, Any

from arc.arc_utils import ArcTask
from nodes import OperatorNode, RelationshipNode, Constant, InputNode, OutputNode, Node
from port_graphs import Edge, PortGraph
from operator_primitives import ConstructorOperator
from relationship_primitives import NotEquals
from synthesis.unified_value_network import UnifiedValueNetwork
from synthesis.arc_specific.catalog_class_structure import CLASS_COMPONENTS_CATALOG
from synthesis.design_time_models import PerceivedValue, WildCardNode


class PerceptionModel:

    __global_id = 0

    @staticmethod
    def get_next_id():
        PerceptionModel.__global_id += 1
        return PerceptionModel.__global_id

    def __init__(self,
                 perception_function_or_functions: Union[Callable, List[Callable]], bg_color=1,
                 applicable_function=lambda challenge: True):
        self.id = PerceptionModel.get_next_id()
        self.perception_functions = perception_function_or_functions if isinstance(perception_function_or_functions,
                                                                                   list) else [
            perception_function_or_functions]
        self.applicable_function = applicable_function
        self.bg_color = bg_color
        self.results_by_case = None
        self.score = None
        #
        # # Store centrally via the context
        # context.perception_models.append(self)

    def applicable(self, challenge: ArcTask) -> bool:
        return self.applicable_function(challenge)

    def evaluated(self) -> bool:
        return self.score is not None

    def to_string(self) -> str:
        return ", ".join([f.__name__ if isinstance(f, Callable) else str(f) for f in self.perception_functions]) + \
            (f" (bg_color={self.bg_color})" if self.bg_color is not None else "")


class TraceGraphInstance:

    # 1. perceived output id - the output id from which this trace is rooted and evaluates to.
    # Not needed but useful for debugging
    # 2. fork indexes - this is how you determine if two graph instances can be merged when sibling trace graphs
    # are merged. The fork indexes include the id of every forking (constructor, operator) node you have visited
    # and the specific index of THIS INSTANCE. NOTE the index (value side) MUST be unique to this graph instance
    # this is a good argument for using a custom collection rather than a list to hold on to these.
    # 3. state - graph state at this point in the search, same pattern as the interpreter.
    def __init__(self, root_id: int, perceived_output_id: int, fork_indexes: Dict[int, int],
                 state: Dict[int, Any], generic_index = None, abstract_to_concrete_starter = None):
        self.root_id = root_id
        self.perceived_output_id = perceived_output_id
        self.fork_indexes = fork_indexes
        self.state = state
        self.__generic_index = generic_index
        # Inputs only
        self.__concrete_to_abstract_lookup: Dict[int, int] = {} if abstract_to_concrete_starter is None else {v:k for k, v in abstract_to_concrete_starter.items()}
        self.__abstract_to_concrete_lookup: Dict[int, int] = {} if abstract_to_concrete_starter is None else abstract_to_concrete_starter

    def get_abstract_from_concrete(self, concrete: int) -> Optional[int]:
        return self.__concrete_to_abstract_lookup.get(concrete)

    def get_concrete_from_abstract(self, abstract: int) -> Optional[int]:
        return self.__abstract_to_concrete_lookup.get(abstract)

    def get_concrete_nodes_included(self):
        return list(self.__concrete_to_abstract_lookup.keys())

    def contains_concrete_node(self, concrete: int) -> bool:
        return concrete in self.__concrete_to_abstract_lookup

    def put_abstract_concrete(self, abstract: int, concrete: int):
        if abstract in self.__abstract_to_concrete_lookup:
            raise Exception(f"Abstract {abstract} already exists")
        self.__abstract_to_concrete_lookup[abstract] = concrete
        if concrete in self.__concrete_to_abstract_lookup:
            raise Exception(f"Concrete {concrete} already exists")
        self.__concrete_to_abstract_lookup[concrete] = abstract
        if len(self.__abstract_to_concrete_lookup) != len(self.__concrete_to_abstract_lookup):
            raise Exception("Mismatch between abstract and concrete")

    def get_abstract_concrete_lookup(self) -> Dict[int, int]:
        return self.__abstract_to_concrete_lookup

    # Merge creating new instance (no side effects)
    # Return NONE if the two are incompatible for merging
    # Abstract duplicates could have been calculated in here but the tuple has use else where.
    # Used to determine how to merge the concrete/abstract lookups - KEEP THE MIN NODE BY ID
    @classmethod
    def validate_and_merge(cls, first: 'TraceGraphInstance', other: 'TraceGraphInstance', abstract_duplicates: Tuple[Tuple]) -> 'TraceGraphInstance':

        # TODO validate state values are not different for same ids that are not Wildcards... in debug mode

        root_id = first.root_id
        if other.root_id != root_id:
            raise Exception(f"Root id {root_id} does not match {other.root_id}")

        percept_id = first.perceived_output_id
        if other.perceived_output_id != percept_id:
            raise Exception(f"Perceived output id {other.perceived_output_id} does not match")

        forked_indexes = first.fork_indexes | other.fork_indexes
        state = first.state | other.state

        merged_instance = cls(root_id, percept_id, forked_indexes, state)

        merged_instance.__abstract_to_concrete_lookup = first.__abstract_to_concrete_lookup | other.__abstract_to_concrete_lookup

        # remove the MAX duplicate because we only keep the min - IF THEY ARE ACTUALLY DIFFERENT
        for duplicate_abstract_ids in abstract_duplicates:
            if len(set(duplicate_abstract_ids)) == len(duplicate_abstract_ids):
                abstract_id_to_remove = max(duplicate_abstract_ids)
                del merged_instance.__abstract_to_concrete_lookup[abstract_id_to_remove]

        # Create concrete to abstract by reversing the abstract to concrete - remember the concretes are where the duplicates where
        merged_instance.__concrete_to_abstract_lookup = {v: k for k, v in merged_instance.__abstract_to_concrete_lookup.items()}

        if len(merged_instance.__concrete_to_abstract_lookup) != len(merged_instance.__abstract_to_concrete_lookup):
            raise Exception("Mismatch between abstract and concrete after merging")

        return merged_instance

    def copy_and_reassign_abstract_ids(self, old_to_new_id_mappings: Dict[int, int]) -> 'TraceGraphInstance':
        new_instance = copy(self)

        new_state = {}
        for k, v in new_instance.state.items():
            new_k = old_to_new_id_mappings.get(k, k)
            new_state[new_k] = v

        new_instance.state = new_state

        new_c_to_a = {}
        for k, v in new_instance.__concrete_to_abstract_lookup.items():
            new_v = old_to_new_id_mappings.get(v, v)
            new_c_to_a[k] = new_v
        new_instance.__concrete_to_abstract_lookup = new_c_to_a

        new_a_to_c = {}
        for k, v in new_instance.__abstract_to_concrete_lookup.items():
            new_k = old_to_new_id_mappings.get(k, k)
            new_a_to_c[new_k] = v
        new_instance.__abstract_to_concrete_lookup = new_a_to_c

        return new_instance

    def __copy__(self):
        copy_instance = TraceGraphInstance(self.root_id, self.perceived_output_id, dict(self.fork_indexes), dict(self.state))
        copy_instance.__abstract_to_concrete_lookup = dict(self.__abstract_to_concrete_lookup)
        copy_instance.__concrete_to_abstract_lookup = dict(self.__concrete_to_abstract_lookup)
        return copy_instance


class TraceEvaluation:
    def __init__(self,
                 program_state: 'ProgramState',
                 trace: 'ProgramTrace',
                 initial_instances: Dict[int, List[TraceGraphInstance]],
                 evaluation_context: 'EvaluationContext'):
        self.trace_id = trace.id
        self.trace_root_type = trace.root_value_type
        self.trace_prior = trace.prior
        self.keying_node_id = trace.starting_abstract_id
        keying_node = trace.graph.get_node_by_id(self.keying_node_id)
        if keying_node is None or not isinstance(keying_node, (InputNode, OutputNode)):
            raise Exception("Bad keying node assumption")
        self.keyed_by_output = isinstance(keying_node, OutputNode)
        self.perception_model_in_id = evaluation_context.perception_in_id
        self.perception_model_out_id = evaluation_context.perception_out_id
        self.instance_ids = set(initial_instances.keys())
        self.keying_perception_model = self.perception_model_out_id if self.keyed_by_output else self.perception_model_in_id
        self.full = False
        self.eval_temp_likelihood = 1
        self.eval_posterior = 1

        # STORE
        program_state.get_evaluation_matrix().record_evaluation(self)

        self.update_instances(program_state, initial_instances, evaluation_context)


    #     # Boolean which "cache" info which values on the instance key are expected to be null
    #     self.instance_key_use_perception_in = not self.keyed_by_output or len(trace.abstract_input_nodes) > 1 # TODO confirm...
    #     self.instance_key_use_perception_out = False # self.keyed_by_output # TODO investigate...
    #
    # # # Default key is: (trace_id, perceived_value_id, Optional[Perception In Id], Optional[Perception Out Id])
    # # def __generate_instance_key(self, perceived_value_id) -> Tuple[int, int, Optional[int], Optional[int]]:
    # #     return (
    # #         self.trace_id,
    # #         perceived_value_id,
    # #         self.perception_model_in_id if self.instance_key_use_perception_in else None,
    # #         self.perception_model_out_id if self.instance_key_use_perception_out else None
    # #     )

        ## TODO cleanup -> these are instance depdenent attrs which were originally stored on the trace itself
        self.eval_case_count = len(
            set(program_state.get_perception_xref().get_perceived_value(out_id).case_index for out_id in initial_instances.keys()))

        self.eval_concrete_concept = None

    def get_key(self) -> Tuple[int, int, int]:
        return self.trace_id, self.perception_model_in_id, self.perception_model_out_id

    def __evaluate_full(self, program_state: 'ProgramState', evaluation_context: 'EvaluationContext') -> None:

        # Only evaluate full if not already true
        if not self.full:
            if evaluation_context.perceived_subset is None:
                # If all parent traces are full and the synthesis args used DO NOT filter the perceived values evaluated, then full is true
                original_trace = program_state.get_trace_graph().get_trace(self.trace_id)
                if (len(original_trace.parent_ids) == 0 or
                        all(program_state.get_evaluation_matrix().get_evaluation(parent_id, evaluation_context)
                            for parent_id in original_trace.parent_ids)):
                    self.full = True

                # (Unlikely case) Otherwise check if the current instances cover all perceived values
                else:
                    all_perceived_values = program_state.get_uvn().get_perceived_values(self.keying_perception_model, self.trace_root_type, self.keyed_by_output)
                    trace_specific_instances: Dict[int, List[TraceGraphInstance]] = self.__get_trace_eval_instances(program_state)
                    self.full = len(all_perceived_values) == len(trace_specific_instances)

    # TODO need to account for perception models in play in the instances! see commented out idea above
    def update_instances(self, program_state: 'ProgramState',
                         new_instances: Dict[int, List[TraceGraphInstance]],
                         evaluation_context: 'EvaluationContext') -> None:
        trace_specific_instances: Dict[int, List[TraceGraphInstance]] = self.__get_trace_eval_instances(program_state)
        for perceived_id, instance_list in new_instances.items():
            self.instance_ids.add(perceived_id)
            if perceived_id in trace_specific_instances:
                trace_specific_instances[perceived_id].extend(instance_list)
            else:
                trace_specific_instances[perceived_id] = list(instance_list)

        # Always reevaluate the posterior
        self.__reevaluate_posterior(program_state, trace_specific_instances, evaluation_context)

        # Always reevaluate the full flag unless it is already true
        self.__evaluate_full(program_state, evaluation_context)

    def get_instances(self, program_state: 'ProgramState',
                      outer_id_subset: Optional[Set[int]] = None) -> Dict[int, List[TraceGraphInstance]]:
        trace_specific_instances: Dict[int, List[TraceGraphInstance]] = self.__get_trace_eval_instances(program_state)
        return {outer_id: trace_specific_instances[outer_id]
                for outer_id in self.instance_ids
                if outer_id_subset is None or outer_id in outer_id_subset}

    def __get_trace_eval_instances(self, program_state: 'ProgramState') -> Dict[int, List[TraceGraphInstance]]:
        # TODO this has problems - not keyed by anything related to this eval or its perception models
        return program_state.get_evaluation_matrix().get_trace_instances_by_trace_id(self.trace_id)


    def __reevaluate_posterior(self, program_state: 'ProgramState',
                               instances: Dict[int, List[TraceGraphInstance]],
                               evaluation_context: 'EvaluationContext') -> None:
        self.eval_temp_likelihood = 1

        trace = program_state.get_trace_graph().get_trace(self.trace_id)
        output_style = len(trace.graph.get_nodes_by_type(OutputNode.__name__)) > 0

        # Note this is the full set if not partial eval

        keying_perception_model = self.perception_model_out_id if output_style else self.perception_model_in_id
        all_perceived_values = program_state.get_perception_xref().get_all_perceived_values(keying_perception_model)

        coverage_by_case: DefaultDict[int, Set[int]] = defaultdict(set)
        sub_set_by_case: DefaultDict[int, Set[PerceivedValue]] = defaultdict(set)

        for perceived_value in all_perceived_values:
            if (perceived_value.output == output_style and
                    (evaluation_context.perceived_subset is None or
                     perceived_value.id in evaluation_context.perceived_subset)):
                sub_set_by_case[perceived_value.case_index].add(perceived_value)
                if perceived_value.id in instances:
                    coverage_by_case[perceived_value.case_index].add(perceived_value.id)

        for case_index, case_subset in sub_set_by_case.items():
            # NOTE IMPLICITLY REQUIRING COVERAGE ACROSS ALL CASES
            factor = coverage_by_case[case_index]
            sub_set_size = len(case_subset)
            self.eval_temp_likelihood *= (len(factor) / sub_set_size) if sub_set_size > 0 else 0

        if trace.graph.has_generics():
            # TODO generic likelihood is not implemented for partial evals
            # multiply the fraction of generic cases represented in each challenge case - multiply with original likelihood
            # ... for each parent trace ... map parent trace to cases. Reverse that to dict of int,list. Then get fractions
            parent_ids_to_cases: Dict[int, Set[int]] = defaultdict(set)
            for parent_id in trace.parent_ids:
                parent_trace = program_state.get_trace_graph().get_trace(parent_id)
                for out_id in parent_trace.graph_instances(program_state, evaluation_context).keys():
                    case = program_state.get_perception_xref().get_perceived_value(out_id).case_index
                    parent_ids_to_cases[parent_id].add(case)

            # IF ALL GENERICS ONLY EXIST IN A SINGLE CASE, HARD CODE 0 POSTERIOR - NOT REAL GENERIC
            # TODO hard coded - need to make smarter... If only one generic index has more than one hit, disqualify it - too unlikely
            generic_indexes_with_redundant_cases = sum(
                1 if len(cases_per_parent_trace) > 1 else 0 for cases_per_parent_trace in parent_ids_to_cases.values())
            if generic_indexes_with_redundant_cases < 2:
                self.eval_temp_likelihood = 0
            else:
                generic_likelihood = 1
                for cases_covered_by_parent_trace in parent_ids_to_cases.values():
                    generic_likelihood *= len(cases_covered_by_parent_trace) / program_state.get_total_case_count()
                # TODO generic likelihood??
                # self.temp_likelihood *= generic_likelihood

        self.eval_posterior = trace.prior * self.eval_temp_likelihood


class ProgramTrace:

    __global_id = 0

    @staticmethod
    def __get_next_id():
        ProgramTrace.__global_id += 1
        return ProgramTrace.__global_id

    # TODO for efficiency add some args that describe - are we focused on a wildcard? are we focused on a fork?
    #  The next action search code is unnecessarily computing the answers to these questions over and over
    # TODO fix constructor to inherit from parents somehow
    def __init__(self, parent_traces: List['ProgramTrace'], graph: PortGraph, focus_node_id: int, depth: int,
                 terminal: bool, latest_fork: Optional[int], latest_fork_index: Optional[int], root_value_type: Type,
                 program_state: 'ProgramState',
                 search_direction_forward=False,
                 predicate_mode=False):
        self.id = ProgramTrace.__get_next_id()
        self.parent_ids = [t.id for t in parent_traces]
        focus_node = graph.get_node_by_id(focus_node_id)
        if len(parent_traces) == 0:
            if isinstance(focus_node, WildCardNode):
                all_node_ids = graph.get_nodes_by_id().keys()
                if len(all_node_ids) > 2:
                    raise Exception("Unexpected")
                abstract_root = min(filter(lambda n: n != focus_node_id, all_node_ids))
            elif isinstance(focus_node, (InputNode, OutputNode)):
                abstract_root = focus_node.id
            else:
                raise Exception("Unexpected")
        else:
            abstract_root = parent_traces[0].starting_abstract_id
        self.starting_abstract_id = abstract_root
        self.predicate_mode = predicate_mode if len(parent_traces) == 0 else parent_traces[0].predicate_mode
        if len(parent_traces) > 1:
            if not all(self.starting_abstract_id == t.starting_abstract_id for t in parent_traces):
                raise Exception("Parent traces must start from same abstract id")
            if not all(self.predicate_mode == t.predicate_mode for t in parent_traces):
                raise Exception("Parent traces must all have same predicate mode")

        self.graph = graph
        self.search_direction_forward = search_direction_forward
        # try:
        #     self.generic_dfs_code = get_min_dfs_code(graph, DfsCodeSchema.GENERIC)
        #     self.generic_dfs_code_str = self.generic_dfs_code.get_string()
        # except KeyError:
        #     print("Generic bug")
        self.original_dfs_code_if_generic = None
        self.focus_node_id = focus_node_id
        self.depth = depth
        self.terminal = terminal
        self.latest_fork = latest_fork
        self.latest_fork_index = latest_fork_index
        self.root_value_type = root_value_type

        self.refocused = any(parent.refocused for parent in parent_traces)

        self.wildcard_count = len(list(n for n in graph.get_nodes_by_id().values() if isinstance(n, WildCardNode)))
        self.abstract_input_nodes = list(map(lambda n: n.id,
                                             filter(lambda n: isinstance(n, InputNode),
                                                    graph.get_nodes_by_id().values())))
        self.prior = (1 / sum(self.__prior_node_weights(n) for n in self.graph.get_nodes_by_id())) * (1 / (depth + 1))

        # Next directive metadata TODO maybe store this centrally somewhere else...? Its fine for now.

        # self.__possible_extension_directives: Set['DirectiveType'] = set() # TODO make dict with estimated lookahead posteriors
        # self.__evaluated_extension_directives: Set['DirectiveType'] = set()

        self.__possible_extension_directives: Dict[Tuple[Type['SynthesisDirective'], Type[Node]], Callable[['ProgramTrace'], 'SyntheisDirective']] = {}
        self.__evaluated_extension_directives: Set[Tuple[Type['SynthesisDirective'], Type[Node]]] = set()

        self.__can_merge_as_join_focus = False

        # Case predicate trace forward
        if self.search_direction_forward:

            # Case: Extend up node - has wildcards and focus node is
            if isinstance(focus_node, WildCardNode):
                self.__possible_extension_directives[(ExtendUpNodeDirective, RelationshipNode)] = ExtendUpNodeDirective

            # Case: Extend up edge - has wildcards and focus node is not a wildcard or operator
            elif not isinstance(focus_node, OperatorNode):
                self.__possible_extension_directives[(ExtendUpEdgeDirective, Edge)] = ExtendUpEdgeDirective

            # Case: Join FOCUS - has wildcards and focus node is an operator (it is the central node whose children are merged over)
            elif isinstance(focus_node, OperatorNode):
                raise NotImplementedError()

        # Case operator or predicate traces
        elif self.wildcard_count > 0:

            # Case extend down predicate mode: # TODO this should eventually be removed and all extend down options should be valid
            if self.predicate_mode:

                if isinstance(focus_node, WildCardNode):
                    self.__possible_extension_directives[(ExtendDownConstantDirective, Constant)] = ExtendDownConstantDirective
                    self.__possible_extension_directives[(ExtendDownInputDirective, InputNode)] = ExtendDownInputDirective

                elif isinstance(focus_node, OperatorNode):
                    raise NotImplementedError()

            else:
                # Case: Extend down - has wildcards and focus node is a wildcard
                if isinstance(focus_node, WildCardNode):
                    for directive_class_and_node, directive_fn in EXTENSION_DOWN_TEMPLATES.items():
                        if directive_class_and_node[1] != ConstructorOperator and issubclass(directive_class_and_node[1], OperatorNode):
                            operator_node: OperatorNode = directive_class_and_node[1]()
                            if operator_node.output_type == focus_node.value_type:
                                self.__possible_extension_directives[directive_class_and_node] = directive_fn
                        else:
                            self.__possible_extension_directives[directive_class_and_node] = directive_fn
                # Case: Join FOCUS - has wildcards and focus node is an operator (it is the central node whose children are merged over)
                elif isinstance(focus_node, OperatorNode):
                    self.__can_merge_as_join_focus = True

        self.__can_merge = len(self.__possible_extension_directives) == 0 and self.terminal

    @staticmethod
    def __prior_node_weights(node: Node):
        if isinstance(node, (InputNode, OutputNode)):
            # Slightly prefer input/output nodes
            return 1.0
        elif isinstance(node, NotEquals):
            # Tax NotEquals nodes
            return 2.0
        return 1.01

    def __get_trace_eval_key(self, evaluation_context: 'EvaluationContext'):
        return self.id, evaluation_context.perception_in_id, evaluation_context.perception_out_id

    def get_trace_eval(self, program_state: 'ProgramState', evaluation_context: 'EvaluationContext') -> TraceEvaluation:
        key = self.__get_trace_eval_key(evaluation_context)
        return program_state.get_evaluation_matrix().get_evaluation_by_key(key)

    def graph_instances(self, program_state: 'ProgramState', evaluation_context: 'EvaluationContext') -> Dict[int, List[TraceGraphInstance]]:
        return self.get_trace_eval(program_state, evaluation_context).get_instances(program_state, evaluation_context.perceived_subset)

    def update_instances(self, program_state: 'ProgramState',
                         instances: Dict[int, List[TraceGraphInstance]],
                         evaluation_context: 'EvaluationContext') -> None:
        key = self.__get_trace_eval_key(evaluation_context)
        trace_eval = program_state.get_evaluation_matrix().get_evaluation_by_key(key)
        if trace_eval is None:
            TraceEvaluation(program_state, self, instances, evaluation_context)
        else:
            trace_eval.update_instances(program_state, instances, evaluation_context)

    def record_directive(self, directive: 'SynthesisDirective') -> None:
        key = (directive.__class__, directive.next_dsl_member)
        self.__evaluated_extension_directives.add(key)

    def can_merge(self) -> bool:
        return self.__can_merge

    def can_merge_as_join_focus(self) -> bool:
        return self.__can_merge_as_join_focus

    def can_extend(self) -> bool:
        return len(self.__evaluated_extension_directives) < len(self.__possible_extension_directives)

    def get_remaining_extension_directives(self) -> List['SynthesisDirective']:
        remaining_directives = []
        for directive_key, directive_builder in self.__possible_extension_directives.items():
            if directive_key not in self.__evaluated_extension_directives:
                dir = directive_builder(self)
                remaining_directives.append(dir)
        return remaining_directives


class PerceptionXref:

    def __init__(self):
        self.__perception_models: Dict[int, 'PerceptionModel'] = {}
        self.__perceived_values: Dict[int, PerceivedValue] = {}
        self.__perceived_values_to_perception_models: DefaultDict[int, Set[int]] = defaultdict(set)
        self.__perception_models_to_perceived_values: DefaultDict[int, Set[int]] = defaultdict(set)

        self.__perception_models_to_test_perceived_values: DefaultDict[int, Set[int]] = defaultdict(set)

    def record_perception(self, perception_model: 'PerceptionModel',
                          perceived_values: List[PerceivedValue],
                          test_perceived_values: List[PerceivedValue],):
        self.__perception_models.setdefault(perception_model.id, perception_model)

        for perceived_value in perceived_values:
            self.__perceived_values.setdefault(perceived_value.id, perceived_value)
            self.__perceived_values_to_perception_models[perceived_value.id].add(perception_model.id)
            self.__perception_models_to_perceived_values[perception_model.id].add(perceived_value.id)

        for test_perceived_value in test_perceived_values:
            self.__perceived_values.setdefault(test_perceived_value.id, test_perceived_value)
            self.__perception_models_to_test_perceived_values[perception_model.id].add(test_perceived_value.id)

    def get_perceived_value(self, perceived_value_id: int) -> PerceivedValue:
        return self.__perceived_values[perceived_value_id]

    def get_perception_model(self, perception_model_id: int) -> 'PerceptionModel':
        return self.__perception_models[perception_model_id]

    def get_all_perception_models(self) -> List['PerceptionModel']:
        return list(self.__perception_models.values())

    def get_all_perceived_values(self, perception_model_id: int) -> List[PerceivedValue]:
        return [self.__perceived_values[p_id] for p_id in self.__perception_models_to_perceived_values[perception_model_id]]

    def get_all_perceived_test_values(self, perception_model_id: int) -> List[PerceivedValue]:
        return [self.__perceived_values[p_id] for p_id in
                self.__perception_models_to_test_perceived_values[perception_model_id]]

class TraceGraph:

    def __init__(self):
        self.__trace_index: Dict[int, ProgramTrace] = {}
        # Unified cache: Key is (tuple of trace ids, 'DirectiveKey' instance), Value is the resulting list of trace ids
        # (Parents, Directive) -> Children
        self.__unified_trace_cache: Dict[Tuple[Tuple[int, ...], 'DirectiveKey'], List[int]] = {}
        # Reverse cache: Key is trace_id, Value is tuple 'DirectiveKey' and parent traces that generated this trace
        # Child -> (Directive, Parents)
        self.__trace_directive_reverse_cache: Dict[int, Tuple['DirectiveKey', Tuple[int, ...]]] = {}

        self.__generic_traces: DefaultDict[Tuple[int, str], List[ProgramTrace]] = defaultdict(list)
        self.__merge_failures: Set[Tuple[int, int]] = set()

        # NEW CACHES
        self.__new_unified_trace_cache: Dict[Tuple, List[int]] = {}

        # EXTENSION METADATA
        # Dict of trace id (must have wildcard extending down) to graph instances where the branch below this wildcard has completely concretized.
        self.__wildcard_trace_concrete_extensions: DefaultDict[int, Set[int]] = defaultdict(set)

    def add_root_trace(self, root_trace: ProgramTrace) -> None:
        if len(root_trace.parent_ids) > 0:
            raise Exception("Root trace cannot have parents")
        self.__trace_index[root_trace.id] = root_trace

    def get_trace(self, trace_id: int) -> ProgramTrace:
        return self.__trace_index[trace_id]

    def get_or_create_traces(self, parent_trace_ids: Tuple[int, ...], directive_key: 'DirectiveKey',
                            trace_supplier: Callable[[], List[ProgramTrace]]) -> List[ProgramTrace]:
        cache_key = (parent_trace_ids, directive_key)
        existing_traces = self.__unified_trace_cache.get(cache_key)
        if existing_traces is not None:
            return [self.__trace_index[trace_id] for trace_id in existing_traces]

        new_traces: List[ProgramTrace] = trace_supplier()

        main_trace_id = parent_trace_ids[0]
        basic_synthesis_action = DIRECTIVE_TYPE_TO_BASIC_ACTION.get(directive_key.directive_type)
        dsl_resource = directive_key.get_dsl_resource()
        dependent_ids = tuple(parent_trace_ids[1:]) if len(parent_trace_ids) > 1 else None
        new_cache_key = (main_trace_id, basic_synthesis_action, dsl_resource, dependent_ids)
        self.__new_unified_trace_cache[new_cache_key] = new_traces

        self.__unified_trace_cache[cache_key] = [trace.id for trace in new_traces]
        for new_trace in new_traces:
            self.__trace_index[new_trace.id] = new_trace
            self.__trace_directive_reverse_cache[new_trace.id] = directive_key, parent_trace_ids

        return new_traces

    def contains_trace(self, parent_trace_ids: Tuple[int, ...], directive_key: 'DirectiveKey'):
        return (parent_trace_ids, directive_key) in self.__unified_trace_cache

    def synthesis_action_already_run(self, trace_id: int, basic_synthesis_action,
                                     dsl_member: Optional[Union[Type[Node],Type[Edge]]] = None,
                                     parameterized_traces: Optional[Tuple[int, int]] = None) -> bool:
        new_cache_key = (trace_id, basic_synthesis_action, dsl_member, parameterized_traces)
        return new_cache_key in self.__new_unified_trace_cache

    def get_synthesis_action_result(self, trace_id: int, basic_synthesis_action,
                                    dsl_member: Optional[Union[Type[Node],Type[Edge]]] = None,
                                    parameterized_traces: Optional[Tuple[int, int]] = None) -> List[ProgramTrace]:
        new_cache_key = (trace_id, basic_synthesis_action, dsl_member, parameterized_traces)
        return self.__new_unified_trace_cache.get(new_cache_key, [])

    def record_directive(self, main_trace_id: int, basic_synthesis_action, dsl_resource, dependent_ids,
                         trace_results: List[ProgramTrace]):
        new_cache_key = (main_trace_id, basic_synthesis_action, dsl_resource, dependent_ids)
        self.__new_unified_trace_cache[new_cache_key] = trace_results

    def get_all_traces(self) -> List['ProgramTrace']:
        return list(self.__trace_index.values())

    def record_generic_trace(self, key: Tuple[str, int], trace: 'ProgramTrace'):
        self.__generic_traces[key].append(trace)

    def get_generic_traces(self, key: Tuple[str, int]) -> List['ProgramTrace']:
        return self.__generic_traces.get(key, [])

    def check_existing_merge(self, join_ids_ordered: Tuple[int, int], directive_type: 'DirectiveType'):
        return (any(parent_tuple == join_ids_ordered and directive_key.directive_type == directive_type
                   for parent_tuple, directive_key in self.__unified_trace_cache.keys()) or
                join_ids_ordered in self.__merge_failures)

    def record_merge_failure(self, ordered_ids: Tuple[int, int]):
        self.__merge_failures.add(ordered_ids)

    def get_trace_directive_reverse_cache(self) -> Dict[int, Tuple['DirectiveKey', Tuple[int, ...]]]:
        return self.__trace_directive_reverse_cache

    def get_program_index(self, root_value_type: type) -> Dict[int, 'ProgramTrace']:
        # TODO implement
        raise NotImplementedError()

    def record_concretized_branch(self, parent_trace_id: int, resulting_concrete_trace_id: int):
        self.__wildcard_trace_concrete_extensions[parent_trace_id].add(resulting_concrete_trace_id)

    def get_concretized_children(self, trace_id: int):
        return self.__wildcard_trace_concrete_extensions[trace_id]

class EvaluationContext:

    def __init__(self, perception_in_id: int, perception_out_id: int,
                 predicate_mode: bool = False,
                 perceived_subset: Optional[Set[int]] = None):
        self.perception_in_id = perception_in_id
        self.perception_out_id = perception_out_id
        self.predicate_mode = predicate_mode # TODO this belongs somewhere else - it is a parameter for synthesis not eval
        self.perceived_subset = perceived_subset
        self.allow_extend_down_constants: bool = False
        self.allow_extend_down_inputs: bool = False
        self.allow_extend_down_operators: bool = False
        self.allow_extend_down_constructors: bool = False
        self.allow_extend_down_outputs: bool = False


class EvaluationMatrix:

    def __init__(self):
        # Dict { trace eval key (trace id, perc in id, perc out id): Trace Eval}
        self.__trace_evaluations: Dict[Tuple[int, int, int], 'TraceEvaluation'] = {}
        # Dict {int trace id: dict {int perceived value id: List[TraceInstances]}} # IMPORTANT inner dict is NOT default
        self.__trace_instances: DefaultDict[int, Dict[int, List['TraceGraphInstance']]] = defaultdict(dict)

    def record_evaluation(self, evaluation: 'TraceEvaluation'):
        self.__trace_evaluations[evaluation.get_key()] = evaluation

    def get_evaluation(self, trace_id: int, evaluation_context: EvaluationContext) -> 'TraceEvaluation':
        key = (trace_id, evaluation_context.perception_in_id, evaluation_context.perception_out_id)
        return self.get_evaluation_by_key(key)

    def get_evaluation_by_key(self, key: Tuple[int, int, int]) -> 'TraceEvaluation':
        return self.__trace_evaluations.get(key)

    def get_trace_instances_by_trace_id(self, trace_id: int) -> Dict[int, List['TraceGraphInstance']]:
        return self.__trace_instances[trace_id]


class ProgramState:

    def __init__(self, challenge: ArcTask):

        # Challenge metadata
        self.__challenge: ArcTask = challenge
        self.__total_case_count: int = len(challenge.train)

        # Perception & value stores
        self.__perception_xref: PerceptionXref = PerceptionXref()
        self.__unified_value_network: UnifiedValueNetwork = UnifiedValueNetwork(CLASS_COMPONENTS_CATALOG)

        # Program storage
        self.__trace_graph: TraceGraph = TraceGraph()

        # Evaluation storage
        self.__evaluation_matrix = EvaluationMatrix()

    def get_challenge(self) -> ArcTask:
        return self.__challenge

    def get_total_case_count(self):
        return self.__total_case_count

    def get_perception_xref(self) -> PerceptionXref:
        return self.__perception_xref

    def get_uvn(self) -> UnifiedValueNetwork:
        return self.__unified_value_network

    def get_trace_graph(self) -> TraceGraph:
        return self.__trace_graph

    def get_evaluation_matrix(self) -> EvaluationMatrix:
        return self.__evaluation_matrix


# NOTE BAD PRACTICE - BUT THESE NEED TO BE IMPORTED DOWN HERE TO AVOID CIRCULAR IMPORTS
from synthesis.synthesis_engine import (EXTENSION_DOWN_TEMPLATES, ExtendUpNodeDirective, ExtendUpEdgeDirective,
                                            ExtendDownConstantDirective, ExtendDownInputDirective)
from synthesis.synthesis_models import DIRECTIVE_TYPE_TO_BASIC_ACTION