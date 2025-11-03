from typing import Any

from nodes import RelationshipNode
from base_types import DslSet


class Equals(RelationshipNode):

    def __init__(self):
        super().__init__([Any, Any])

    def apply(self, object_1: Any, object_2: Any):
        return object_1 == object_2

    def is_symmetrical(self) -> bool:
        return True


class NotEquals(RelationshipNode):

    def __init__(self):
        super().__init__([Any, Any])

    def apply(self, object_1: Any, object_2: Any):
        return object_1 != object_2

    def is_symmetrical(self) -> bool:
        return True


class LessThan(RelationshipNode):

    def __init__(self):
        super().__init__([Any, Any])

    def apply(self, object_1: Any, object_2: Any):
        return object_1 < object_2


class LessThanOrEqual(RelationshipNode):
    def __init__(self):
        super().__init__([Any, Any])

    def apply(self, object_1: Any, object_2: Any):
        return object_1 <= object_2


class GreaterThan(RelationshipNode):
    def __init__(self):
        super().__init__([Any, Any])

    def apply(self, object_1: Any, object_2: Any):
        return object_1 > object_2


class GreaterThanOrEqual(RelationshipNode):
    def __init__(self):
        super().__init__([Any, Any])

    def apply(self, object_1: Any, object_2: Any):
        return object_1 >= object_2


class SetContains(RelationshipNode):

    def __init__(self):
        super().__init__([DslSet, Any])

    def apply(self, object_1: DslSet, object_2: Any):
        return object_2 in object_1


class SetNotContains(RelationshipNode):

    def __init__(self):
        super().__init__([DslSet, Any])

    def apply(self, object_1: DslSet, object_2: Any):
        return not SetContains().apply(object_1, object_2)
