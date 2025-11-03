from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Type, Callable

from ordered_set import OrderedSet


class DslConstructor:

    def __init__(self, fn: Callable, args: List[Tuple[str, Type]]):
        self.fn = fn
        self.args = args


class DslType(ABC):

    @classmethod
    @abstractmethod
    def get_constructors(cls) -> List[DslConstructor]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        raise NotImplementedError()


class PerceivedType(DslType):

    # TODO rename to perception "ID"
    @abstractmethod
    def get_perception_function(self):
        raise NotImplementedError()


class DslSet(DslType, OrderedSet):

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return []

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [("size", int)]

    def __hash__(self):
        return hash(frozenset(self)) # TODO optimize


def get_source_port_value(source_node, source_port: Optional[str]):
    if source_port is None or source_port == "":
        return source_node

    # Set source port applied for-each
    if source_port.startswith("*"):
        if not source_port.startswith("*."):
            raise Exception("Invalid set source port")
        individual_source_port = source_port[2:]
        return DslSet(get_source_port_value(set_value, individual_source_port) for set_value in source_node)

    # Set size
    if source_port == "size":
        return len(source_node)

    value = source_node
    path = source_port.split(".")

    for attribute in path:
        value = getattr(value, attribute)

    return value
