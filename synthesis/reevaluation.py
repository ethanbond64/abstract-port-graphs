from copy import copy

from synthesis.design_time_models import WildCardNode
from synthesis.program_state import ProgramTrace
from synthesis.synthesis_models import DirectiveType
from synthesis.synthesis_engine import ExtendUpEdgeDirective, ExtendUpNodeDirective, ForkJoinDirective, \
    LeafJoinDirective, GenericMergeDirective, ExtendDownInputDirective, ExtendDownConstructorDirective, \
    ExtendDownOperatorDirective, ExtendDownConstantDirective
from synthesis.program_state import ProgramState, EvaluationContext


def reevaluate_trace(program_state: ProgramState, trace: ProgramTrace, evaluation_context: EvaluationContext):

    if evaluation_context.perceived_subset is not None:
        raise Exception("Unimplemented")

    # Check if trace is fully evaluated already
    if program_state.get_evaluation_matrix().get_evaluation(trace.id, evaluation_context).full:
        return trace

    # If not, get the trace cache reverse entry
    directive_key, parent_ids = program_state.get_trace_graph().get_trace_directive_reverse_cache().get(trace.id)

    program_index = program_state.get_trace_graph().get_program_index(trace.root_value_type)
    parent_traces = [program_index[parent_id] for parent_id in parent_ids]

    # Recursively reevaluate parents first to ensure they have all eval values for this trace
    # TODO cycle risk!
    for parent_trace in parent_traces:
        reevaluate_trace(program_state, parent_trace, evaluation_context)

    if len(parent_traces) == 0:
        raise Exception("Bad assumption")

    main_parent_trace = parent_traces[0]
    main_parent_focus_node = main_parent_trace.graph.get_node_by_id(main_parent_trace.focus_node_id)

    fn_evaluation_context = evaluation_context
    if directive_key is None:
        raise Exception("Bad assumption")

    elif directive_key.directive_type == DirectiveType.EXTEND_DOWN_CONSTANT:
        directive = ExtendDownConstantDirective(main_parent_trace)
        fn_evaluation_context = copy(fn_evaluation_context)
        fn_evaluation_context.allow_extend_down_constants = True
        fn_evaluation_context.allow_extend_down_inputs = False
        fn_evaluation_context.allow_extend_down_operators = False
        fn_evaluation_context.allow_extend_down_constructors = False
        fn_evaluation_context.allow_extend_down_outputs = False

    elif directive_key.directive_type == DirectiveType.EXTEND_DOWN_INPUT:
        directive = ExtendDownInputDirective(main_parent_trace)
        fn_evaluation_context = copy(fn_evaluation_context)
        fn_evaluation_context.allow_extend_down_constants = False
        fn_evaluation_context.allow_extend_down_inputs = True
        fn_evaluation_context.allow_extend_down_operators = False
        fn_evaluation_context.allow_extend_down_constructors = False
        fn_evaluation_context.allow_extend_down_outputs = False

    elif directive_key.directive_type == DirectiveType.EXTEND_DOWN_CONSTRUCTOR:
        directive = ExtendDownConstructorDirective(main_parent_trace)
        fn_evaluation_context = copy(fn_evaluation_context)
        fn_evaluation_context.allow_extend_down_constants = False
        fn_evaluation_context.allow_extend_down_inputs = False
        fn_evaluation_context.allow_extend_down_operators = False
        fn_evaluation_context.allow_extend_down_constructors = True
        fn_evaluation_context.allow_extend_down_outputs = False

    elif directive_key.directive_type == DirectiveType.EXTEND_DOWN_OPERATOR:
        directive = ExtendDownOperatorDirective(main_parent_trace)
        fn_evaluation_context = copy(fn_evaluation_context)
        fn_evaluation_context.allow_extend_down_constants = False
        fn_evaluation_context.allow_extend_down_inputs = False
        fn_evaluation_context.allow_extend_down_operators = True
        fn_evaluation_context.allow_extend_down_constructors = False
        fn_evaluation_context.allow_extend_down_outputs = False

    elif directive_key.directive_type == DirectiveType.EXTEND_UP_EDGE:
        directive = ExtendUpEdgeDirective(main_parent_trace)

    elif directive_key.directive_type == DirectiveType.EXTEND_UP_NODE:
        if not isinstance(main_parent_focus_node, WildCardNode):
            raise Exception("ExtendUpNode requires WildCardNode focus")
        directive = ExtendUpNodeDirective(main_parent_trace)

    elif directive_key.directive_type == DirectiveType.FORK_JOIN:
        if (parent_traces[0].parent_ids[0] != parent_traces[1].parent_ids[0] or
            len(parent_traces[0].parent_ids[0]) != 1 or
            len(parent_traces[0].parent_ids[1]) != 1):
            raise Exception("Bad assumption")
        join_focus_trace = program_index[parent_traces[0].parent_ids[0]]
        directive = ForkJoinDirective(join_focus_trace, parent_traces[0], parent_traces[1])

    elif directive_key.directive_type == DirectiveType.LEAF_JOIN:
        if len(parent_traces) < 2:
            raise Exception("LeafJoin requires 2 parent traces")
        directive = LeafJoinDirective(parent_traces[0], parent_traces[1])

    elif directive_key.directive_type == DirectiveType.GENERIC_MERGE:
        directive = GenericMergeDirective(parent_traces)

    else:
        raise Exception(f"Unhandled directive type: {directive_key.directive_type}")

    # Run the directive to reevaluate
    directive.run(program_state, fn_evaluation_context)

    return trace
