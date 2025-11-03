from typing import List, Tuple, Optional, Union, Type, Any, Dict

from nodes import Constant, OperatorNode, Node, InputNode
from port_graphs import Edge
from operator_primitives import ConstructorOperator
from synthesis.arc_specific.catalog_class_structure import CLASS_CONSTRUCTOR_CATALOG
from synthesis.arc_specific.catalog_operators import TYPE_TO_OPERATOR, OPERATOR_TO_STRATEGY
from synthesis.arc_specific.catalog_relationships_new import TEMP_RELATIONSHIP_SEARCH_STRATEGIES, RelationshipSearchStrategy
from synthesis.design_time_models import WildCardNode
from synthesis.execution import SynthesisAction
from synthesis.priors import NODE_PRIORS
from synthesis.program_state import ProgramState, EvaluationContext, ProgramTrace
from synthesis.synthesis_engine import SynthesisDirective, ExtendUpEdgeDirective, ExtendUpNodeDirective, \
    ExtendDownOperatorDirective, ForkJoinDirective, ExtendDownConstantDirective, ExtendDownInputDirective, \
    ExtendDownConstructorDirective
from synthesis.synthesis_models import BasicSynthesisActions
from synthesis.synthesis_action_selection import AbstractNodeSoloInstances


def choose_from_all_actions(program_state: ProgramState,
                            evaluation_context: EvaluationContext,
                            program_subset: List[ProgramTrace],
                            target_set: Optional[AbstractNodeSoloInstances] = None) -> SynthesisAction:

    directives_and_metrics = generate_all_possible_directives(program_state, evaluation_context,
                                                              program_subset, target_set)

    # !Important discovery here - It is imperative we fork join every possible combination of terminal branches as
    # soon as they exist meeting that condition. When the branches are free to keep exploring even when they are
    # ready to be merged with sibling branches, combinatorial explosion occurs.
    if any(isinstance(d, ForkJoinDirective) for d, m in directives_and_metrics):
        directives_and_metrics = list(filter(lambda dm: isinstance(dm[0], ForkJoinDirective), directives_and_metrics))
    if len(directives_and_metrics) == 0:
        return None

    best_directive_and_metrics = sorted(directives_and_metrics, key=lambda d_m: d_m[1].product, reverse=True)[0]
    directive = best_directive_and_metrics[0]
    return SynthesisAction([directive], evaluation_context)


def synthesis_question_1_available_actions(program_trace: ProgramTrace) -> BasicSynthesisActions:
    if program_trace.predicate_mode:
        if program_trace.wildcard_count == 0 and program_trace.terminal:
            return BasicSynthesisActions.MERGE_UP
        else:
            if program_trace.search_direction_forward:
                return BasicSynthesisActions.EXTEND_UP
            else:
                return BasicSynthesisActions.EXTEND_DOWN
    else:
        focus_node = program_trace.graph.get_node_by_id(program_trace.focus_node_id)
        if isinstance(focus_node, OperatorNode):
            return BasicSynthesisActions.MERGE_DOWN
        elif isinstance(focus_node, WildCardNode):
            return BasicSynthesisActions.EXTEND_DOWN
    return None

def synthesis_question_2_available_dsl_members(program_trace: ProgramTrace,
                                               basic_action: BasicSynthesisActions) -> Optional[
    List[Union[Type[Node], Type[Edge]]]]:

    if basic_action is None:
        return None

    # Merging does not require an additional dsl member
    if basic_action == BasicSynthesisActions.MERGE_UP or basic_action == BasicSynthesisActions.MERGE_DOWN:
        return None

    focus_node = program_trace.graph.get_node_by_id(program_trace.focus_node_id)
    if basic_action == BasicSynthesisActions.EXTEND_UP:
        if isinstance(focus_node, WildCardNode):
            # TODO cleanup - decide when using relationships and when using strategies - make uniform with operators
            relationship_strategies: List[RelationshipSearchStrategy] = (TEMP_RELATIONSHIP_SEARCH_STRATEGIES[Any] +
                                                                         TEMP_RELATIONSHIP_SEARCH_STRATEGIES[
                                                                             focus_node.value_type])
            relationship_types = []
            for strategy in relationship_strategies:
                relationship_types.append(strategy.relationship.__class__)

            return relationship_types
        else:
            return [Edge]

    elif basic_action == BasicSynthesisActions.EXTEND_DOWN:

        extension_down_types = [
            Constant,
            InputNode
        ]

        if len(CLASS_CONSTRUCTOR_CATALOG.get(focus_node.value_type, [])) > 0:
            extension_down_types.append(ConstructorOperator)

        valid_operators = TYPE_TO_OPERATOR.get(focus_node.value_type, set())

        for operator in valid_operators:
            # TODO cleanup - decide when using operators and when using strategies - make uniform with relationships
            if operator in OPERATOR_TO_STRATEGY:
                extension_down_types.append(operator.__class__)

        return extension_down_types
    else:
        raise Exception("Unknown action")


def synthesis_question_3_available_param_traces(program_state, evaluation_context, main_trace: ProgramTrace,
                                                trace_subset: List[ProgramTrace],
                                                basic_action: BasicSynthesisActions) -> Optional[Tuple[int, ...]]:
    if basic_action not in {BasicSynthesisActions.MERGE_UP, BasicSynthesisActions.MERGE_DOWN}:
        return None

    if basic_action == BasicSynthesisActions.MERGE_UP:
        # Find all fitting predicates for which their merge would result in a smaller set for

        # Merge with focus on target set
        # Merge with focus on non-target set
        return None
        # raise NotImplemented()

    elif basic_action == BasicSynthesisActions.MERGE_DOWN:

        # Find all child traces meeting criteria - diff indexes, fork id == arg trace, complete/terminal
        # trace.root_value_type == root_value_type and
        # trace.can_merge() and
        # trace.latest_fork == join_trace.id
        param_sorted_tuples: List[Tuple[int, int]] = []
        child_trace_candidates: List[ProgramTrace] = list(filter(lambda t: t.latest_fork == main_trace.id and
                                                                           t.terminal and
                                                                           t.root_value_type == main_trace.root_value_type and
                                                                           t.get_trace_eval(program_state, evaluation_context).eval_posterior > 0,
                                                                 trace_subset))
        if len(child_trace_candidates) > 1:
            for index, child_trace_candidate in enumerate(child_trace_candidates):
                for second_child_trace_candidate in child_trace_candidates[index + 1:]:
                    if child_trace_candidate.latest_fork_index != second_child_trace_candidate.latest_fork_index:
                        id_tuple = tuple(sorted((child_trace_candidate.id, second_child_trace_candidate.id)))
                        param_sorted_tuples.append(id_tuple)

        return param_sorted_tuples
    else:
        raise NotImplemented()


BASIC_SYNTHESIS_ACTION_PRIORS: Dict[BasicSynthesisActions, float] = {
    BasicSynthesisActions.EXTEND_DOWN: 0.25,
    BasicSynthesisActions.EXTEND_UP: 0.25,
    BasicSynthesisActions.MERGE_DOWN: 1.0,
    BasicSynthesisActions.MERGE_UP: 0.15,
}

class SynthesisTemplate:

    def __init__(self, action: BasicSynthesisActions,
                 dsl_member: Optional[Union[Type[Node], Type[Edge]]],
                 parameterized_traces: Optional[Tuple[int, int]] = None):
        self.action = action
        self.action_prior = BASIC_SYNTHESIS_ACTION_PRIORS[action]

        if ((action == BasicSynthesisActions.EXTEND_DOWN or
             action == BasicSynthesisActions.EXTEND_UP) and dsl_member is None):
            raise Exception("Invalid")

        if ((action == BasicSynthesisActions.MERGE_UP or
             action == BasicSynthesisActions.MERGE_DOWN) and parameterized_traces is None):
            raise Exception("Invalid")

        self.dsl_member = dsl_member
        self.parameterized_traces = parameterized_traces
        self.dsl_prior = NODE_PRIORS[dsl_member] if dsl_member is not None else 1.0

    def get_priors(self) -> Tuple[float, float]:
        return self.action_prior, self.dsl_prior

    def get_directive_program_prior_and_likelihood(self, program_state: ProgramState, evaluation_context: EvaluationContext,
                                                   program_trace: ProgramTrace) -> Tuple[SynthesisDirective, float, float]:

        if self.action == BasicSynthesisActions.EXTEND_DOWN:

            if program_trace.search_direction_forward:
                raise Exception("Wrong search direction - requires backwards")

            if self.dsl_member == Constant:
                return ExtendDownConstantDirective(program_trace), *self._default_prior_likelihood(program_state, evaluation_context, program_trace)
            elif self.dsl_member == InputNode:
                return ExtendDownInputDirective(program_trace), *self._default_prior_likelihood(program_state, evaluation_context, program_trace)
            elif self.dsl_member == ConstructorOperator:
                return ExtendDownConstructorDirective(program_trace), *self._default_prior_likelihood(program_state, evaluation_context, program_trace)
            elif any(self.dsl_member == o.__class__ for o in OPERATOR_TO_STRATEGY.keys()):
                return ExtendDownOperatorDirective(program_trace, self.dsl_member), *self._default_prior_likelihood(program_state, evaluation_context, program_trace)
            else:
                raise Exception("Invalid")

        elif self.action == BasicSynthesisActions.EXTEND_UP:

            if not program_trace.search_direction_forward:
                raise Exception("Wrong search direction - requires forwards")

            if self.dsl_member == Edge:
                return ExtendUpEdgeDirective(program_trace), *self._default_prior_likelihood(program_state, evaluation_context, program_trace)
            if any(self.dsl_member == s.relationship.__class__ for v in TEMP_RELATIONSHIP_SEARCH_STRATEGIES.values() for s in v):  # or self.dsl_member in OPERATOR_TO_STRATEGY:
                return ExtendUpNodeDirective(program_trace, self.dsl_member), *self._default_prior_likelihood(program_state, evaluation_context, program_trace)

        elif self.action == BasicSynthesisActions.MERGE_DOWN:

            if program_trace.search_direction_forward:
                raise Exception("Wrong search direction - requires backwards")

            if self.parameterized_traces is None:
                raise Exception("Invalid arguments, child traces required for merge")

            merge_trace_1 = program_state.get_trace_graph().get_trace(self.parameterized_traces[0])
            merge_trace_2 = program_state.get_trace_graph().get_trace(self.parameterized_traces[1])

            m_t_1_prior, m_t_1_likelihood = self._default_prior_likelihood(program_state, evaluation_context, merge_trace_1)
            m_t_2_prior, m_t_2_likelihood = self._default_prior_likelihood(program_state, evaluation_context, merge_trace_2)

            avg_prior = (m_t_1_prior + m_t_2_prior) / 2
            avg_likelihood = (m_t_1_likelihood + m_t_2_likelihood) / 2

            likelihood = avg_likelihood if m_t_1_likelihood > 0 and m_t_2_likelihood > 0 else avg_likelihood

            return ForkJoinDirective(program_trace, merge_trace_1, merge_trace_2), avg_prior, likelihood

        elif self.action == BasicSynthesisActions.MERGE_UP:
            raise NotImplementedError()
            # if merge_trace_1 is None:
            #     raise Exception("Invalid arguments, child traces required for merge")
            #
            # return LeafJoinDirective(program_trace, merge_trace_1)

        raise Exception("Invalid")

    def _default_prior_likelihood(self, program_state: ProgramState, evaluation_context: EvaluationContext, program_trace: ProgramTrace) -> Tuple[float, float]:
        return program_trace.prior, program_state.get_evaluation_matrix().get_evaluation(program_trace.id, evaluation_context).eval_temp_likelihood

class SynthesisActionModelParams:

    def __init__(self,
                 action_prior: float,
                 dsl_prior: float,
                 trace_prior: float,
                 eval_likelihood: float,
                 concretized_children_liklihood: Optional[float] = None):
        self.action_prior = action_prior
        self.dsl_prior = dsl_prior
        self.trace_prior = trace_prior
        self.eval_likelihood = eval_likelihood
        self.concretized_children_likelihood = concretized_children_liklihood
        self.product = action_prior * dsl_prior * trace_prior * eval_likelihood
        if concretized_children_liklihood is not None and concretized_children_liklihood > 0:
            self.product *= concretized_children_liklihood
        # self.product = trace_prior * eval_likelihood * dsl_prior

    def get_product(self):
        return self.product

def synthesis_action_all_questions(program_state: ProgramState,
                                   program_subset: List[ProgramTrace],
                                   evaluation_context: EvaluationContext,
                                   target_set: Optional[AbstractNodeSoloInstances],
                                   program_trace: ProgramTrace) -> List[Tuple[SynthesisDirective, SynthesisActionModelParams]]:

    available_action = synthesis_question_1_available_actions(program_trace)
    available_dsl_members = synthesis_question_2_available_dsl_members(program_trace, available_action)
    available_param_traces = synthesis_question_3_available_param_traces(program_state, evaluation_context, program_trace, program_subset, available_action)

    available_synthesis_templates = []
    if available_action in {BasicSynthesisActions.EXTEND_UP, BasicSynthesisActions.EXTEND_DOWN}:
        for dsl_member in available_dsl_members:
            if not program_state.get_trace_graph().synthesis_action_already_run(program_trace.id, available_action,
                                                                                dsl_member=dsl_member):
                available_synthesis_templates.append(SynthesisTemplate(available_action, dsl_member))

    elif available_action == BasicSynthesisActions.MERGE_DOWN:
        if available_param_traces is not None:
            for available_param_tuple in available_param_traces:
                if not program_state.get_trace_graph().synthesis_action_already_run(program_trace.id,
                                                                                    available_action,
                                                                                    dsl_member=None,
                                                                                    parameterized_traces=available_param_tuple):
                    available_synthesis_templates.append(SynthesisTemplate(available_action, None, parameterized_traces=available_param_tuple))

    synthesis_directives_and_metadata: List[Tuple[SynthesisDirective, SynthesisActionModelParams]] = []
    for synthesis_template in available_synthesis_templates:
        directive, program_prior, program_likelihood = synthesis_template.get_directive_program_prior_and_likelihood(program_state, evaluation_context, program_trace)
        directive_priors = synthesis_template.get_priors()
        param_data = SynthesisActionModelParams(*directive_priors, program_prior, program_likelihood)
        synthesis_directives_and_metadata.append((directive, param_data))

    return synthesis_directives_and_metadata

def generate_all_possible_directives(program_state: ProgramState,
                                     evaluation_context: EvaluationContext,
                                     program_subset: List[ProgramTrace],
                                     target_set: Optional[AbstractNodeSoloInstances] = None) -> List[
    Tuple[SynthesisDirective, SynthesisActionModelParams]]:

    all_directives_and_metadata: List[Tuple[SynthesisDirective, SynthesisActionModelParams]] = []
    for program in program_subset:
        directives_and_metadata = synthesis_action_all_questions(program_state, program_subset, evaluation_context,
                                                                 target_set, program)
        all_directives_and_metadata.extend(directives_and_metadata)

    return all_directives_and_metadata
