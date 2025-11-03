from typing import Type, Any

from nodes import Node


# Main class for storing perceived objects distinctly by their source perception.
# Tied to a xref for when multiple perception functions provide the same object.
# KEEP IN DSL FILE DUE TO IMPORTS
class PerceivedValue:
    __global_id = 0

    @staticmethod
    def get_next_id():
        PerceivedValue.__global_id += 1
        return PerceivedValue.__global_id

    def __init__(self, value_type: Type, value: Any, perception_model_id: int, case_index: int,
                 output=False, test=False):
        self.id = PerceivedValue.get_next_id()
        self.value_type = value_type
        self.value = value
        self.perception_model_id = perception_model_id
        self.case_index = case_index
        self.output = output
        self.test = test

    def __eq__(self, __value):
        if __value is None or not isinstance(__value, PerceivedValue):
            return False
        return self.id == __value.id

    def __hash__(self):
        return hash(self.id)


class WildCardNode(Node):

    def __init__(self, value_type: Type):
        super().__init__(0)
        self.value_type = value_type

    def get_outbound_type(self) -> Type:
        return self.value_type
