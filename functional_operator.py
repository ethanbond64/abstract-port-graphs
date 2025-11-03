from typing import List, Type, Any

from nodes import OperatorNode, InputNode, OutputNode
from port_graphs import PortGraph
from interpreter.interpretter_common import GraphIO
from interpreter.interpreter import Interpreter


# For handwritten nodes - Embed the "function" graph building logic in the load_graph method using the injected nodes.
class FunctionalOperator(OperatorNode):

    def __init__(self, input_types: List[Type], output_type: Type):
        super().__init__(input_types, output_type)
        self.graph = None
        self.output_id = None
        self.input_ids = []

    def load_graph(self, graph: PortGraph, output_node: OutputNode, *input_nodes: InputNode) -> PortGraph:
        ...

    def get_graph(self) -> PortGraph:
        if self.graph is None:
            new_graph = PortGraph()

            output_node = OutputNode(value_type=self.output_type)
            self.output_id = output_node.id

            input_nodes = []
            for input_type in self.input_types:
                input_node = InputNode(value_type=input_type, perception_qualifiers=None)
                self.input_ids.append(input_node.id)
                input_nodes.append(input_node)

            self.graph = self.load_graph(new_graph, output_node, *input_nodes)
        return self.graph

    def apply(self, interpreter: Interpreter, *args) -> Any:
        graph = self.get_graph()
        graph_state = GraphIO(graph)
        for i, arg in enumerate(args):
            graph_state.add_input_value(self.input_ids[i], arg)

        output_graph_state = interpreter.evaluate_port_graph(graph, graph_state)

        output_list = output_graph_state.get_output_values()[self.output_id]
        return output_list[0] if len(output_list) > 0 else None

    def primitive(self) -> bool:
        return False
