from abc import ABC, abstractmethod
from typing import Any, List, Dict

from base_types import PerceivedType
from port_graphs import PortGraph


class PerceptionModel(ABC):

    @abstractmethod
    def apply_perception(self, raw_data: Any) -> List[PerceivedType]:
        # See arc directory for subclass implementation
        raise NotImplementedError()


class Program:

    def __init__(self, perception_model: PerceptionModel, graph: PortGraph):
        self.perception_model = perception_model
        self.graph = graph

    # Intended to be overridden.
    def format_output_values(self, output_node_values: Dict[int, List[Any]]) -> Any:
        return output_node_values
