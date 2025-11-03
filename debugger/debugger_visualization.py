import os
import tempfile

import networkx as nx
from pyvis.network import Network

from nodes import Constant, RelationshipNode, OperatorNode, InputNode, OutputNode, InputSetNode
from arc.arc_objects import ArcGraph


def get_node_color(node):
    if isinstance(node, InputNode):
        return "green"
    elif isinstance(node, OutputNode):
        return "red"
    elif isinstance(node, InputSetNode):
        return "purple"
    elif isinstance(node, RelationshipNode):
        return "blue"
    elif isinstance(node, OperatorNode):
        return "orange"
    elif isinstance(node, Constant):
        return "black"

    return "gray"


def generate_graph_html(graph: ArcGraph, active_node: int = None):
    # function_name = "solve_" + challenge
    # graph_function = globals().get(function_name)
    #
    # if graph_function is None:
    #     raise Exception("Graph not found for challenge: " + challenge)
    #
    # graph: ArcGraph = graph_function()

    nx_graph = nx.DiGraph()

    # Add nodes with labels and colors
    node_labels = {}
    for node in graph.graph.get_nodes_by_id().values():
        debug_label = node.get_debug_label()
        debug_label = f"{debug_label}\n" if debug_label and debug_label != "None" else ""
        node_label = f"{type(node).__name__}\n{debug_label}{node.id}"
        color = get_node_color(node)
        size = 10

        if node.id == active_node:
            size = 30

        nx_graph.add_node(node.id, label=node_label, color=color, size=size)
        node_labels[node.id] = node_label

    # Add edges with labels
    edge_labels = {}
    for edge in graph.graph.get_edge_set():
        nx_graph.add_edge(edge.source_node, edge.target_node, label=edge.get_debug_label())
        edge_labels[(edge.source_node, edge.target_node)] = edge.get_debug_label()

    # Compute positions using spring_layout (repulsion-based) with a fixed seed
    # pos = nx.spiral_layout(nx_graph)
    # for node_id, p in pos.items():
    #     # Scale positions as needed; here we multiply by 1000 for visibility
    #     nx_graph.nodes[node_id]['x'] = p[0] * 600
    #     nx_graph.nodes[node_id]['y'] = p[1] * 600
    #     nx_graph.nodes[node_id]['fixed'] = True  # mark positions as fixed

    # Create the Pyvis network, load the nodes with fixed positions, and disable physics
    net = Network(notebook=True, directed=True)
    net.from_nx(nx_graph)
    # net.toggle_physics(False)

    # nx.spring_layout(nx_graph, seed=42)
    # # Create pyvis network
    # net = Network(notebook=True, directed=True)
    # net.from_nx(nx_graph)
    # net.set_options('''{
    #   "physics": {
    #     "repulsion": {
    #       "centralGravity": 0.3,
    #       "springLength": 200,
    #       "springConstant": 0.05,
    #       "nodeDistance": 150,
    #       "damping": 0.09
    #     },
    #     "minVelocity": 0.75
    #   }
    # }''')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        net.show(tmpfile.name)

        return os.path.abspath(tmpfile.name)
