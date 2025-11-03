from arc.arc_objects import ArcObject, EnvironmentShape, ArcEnvironment, ArcGraph
from nodes import InputNode, OutputNode
from port_graphs import PortGraph
from synthesis.synthesis_action_selection import ProgramHypothesis


def draft_program_to_arc_graph(draft_program: ProgramHypothesis):

    env = ArcEnvironment(draft_program.perception_model_in.perception_functions, draft_program.perception_model_in.bg_color)
    arc_program = ArcGraph(env)
    final_graph = arc_program.graph

    ### MERGE MASTER GRAPHS TO SINGLE GRAPH ###
    master_graphs = draft_program.create_master_graphs_from_hypothesis(ArcObject)
    for master_graph in master_graphs:
        final_graph, _ = PortGraph.merge_graphs(master_graph, final_graph, {})

    arc_program.graph = final_graph

    ### MERGE ENV GRAPH WITH FINAL GRAPH ###
    arc_program = prep_final_graph_env_shape_ids(draft_program, arc_program)
    arc_program = clean_final_graph_nodes(draft_program, arc_program)

    return arc_program


def prep_final_graph_env_shape_ids(draft_program: ProgramHypothesis, arc_program: ArcGraph) -> ArcGraph:
    op_dict = draft_program.confirmed_operator_traces[EnvironmentShape]
    todo_temp_best_env_shape_trace = list(op_dict.values())[0]
    env_input_ids = list(n for n in todo_temp_best_env_shape_trace.graph.get_nodes_by_id().values()
                        if isinstance(n, InputNode) and n.value_type == EnvironmentShape)
    env_output_ids = list(n for n in todo_temp_best_env_shape_trace.graph.get_nodes_by_id().values()
                         if isinstance(n, OutputNode) and n.value_type == EnvironmentShape)
    if len(env_output_ids) > 1 or len(env_input_ids) > 1:
        raise Exception("TODO env relies on multiple nodes")
    env_output_id = env_output_ids[0].id
    env_lookup = {env_output_id: arc_program.env.output_matrix_node.id}
    if len(env_input_ids) > 0:
        env_input_id = env_input_ids[0].id
        input_matrix_node = arc_program.env.input_matrix_node()
        arc_program.graph.add_node(input_matrix_node)
        env_lookup[env_input_id] = input_matrix_node.id
    merged_graph, _ = PortGraph.merge_graphs(todo_temp_best_env_shape_trace.graph, arc_program.graph, env_lookup)
    arc_program.graph = merged_graph
    return arc_program


def clean_final_graph_nodes(draft_program: ProgramHypothesis, arc_program: ArcGraph):
    final_graph_nodes_safe = list(arc_program.graph.get_nodes_by_id().values())
    for node in final_graph_nodes_safe:
        if isinstance(node, InputNode) and node.value_type == ArcObject:
            abstract_node = InputNode(value_type=node.value_type,
                perception_qualifiers=set(draft_program.perception_model_in.perception_functions))
            arc_program.graph.replace_node(node, abstract_node, maintain_multiple_edges=True)
        if isinstance(node, OutputNode) and node.value_type == EnvironmentShape:
            arc_program.graph.replace_node_with_new_node_and_inbound_edges(node, arc_program.env.output_matrix_node)
        elif isinstance(node, InputNode) and node.value_type == EnvironmentShape:
            input_env_node = arc_program.env.input_matrix_node()
            arc_program.graph.replace_node(node, input_env_node, maintain_multiple_edges=True)
    return arc_program
