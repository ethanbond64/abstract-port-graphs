from copy import copy
from numbers import Number
from typing import Type, List, Any

from nodes import OperatorNode
from base_types import DslSet


class ConstructorOperator(OperatorNode):

    def __init__(self):
        super().__init__([Type, List[Any]], Type) # TODO custom base type?

    def apply(self, custom_type: Type, *args: List[Any]) -> Type:
        return custom_type(*[copy(arg) for arg in args])


# Not-bijective
class SetRankOperator(OperatorNode):

    def __init__(self, reverse=False):
        super().__init__([DslSet[Any], Any], int)  # TODO really should be set of numbers/numerics
        self.reverse = reverse

    def apply(self, input_set: DslSet, input_object) -> int:
        if input_object not in input_set:
            raise Exception("Cannot rank a non-member of the set")

        return DslSet(sorted(input_set, reverse=self.reverse)).index(input_object)


class ApplyScalarOpToSetOperator(OperatorNode):
    def __init__(self):
        super().__init__([OperatorNode, DslSet[Any]], DslSet[Any])

    def apply(self, inner_operator: OperatorNode, *args):

        # Ensure all iterable args are the same size
        last_size = None
        args_as_lists = []
        for arg in args:
            if isinstance(arg, DslSet):
                if last_size is None:
                    last_size = len(arg)
                elif last_size != len(arg):
                    raise ValueError("Inconsistent size")
                args_as_lists.append(list(arg))
            else:
                args_as_lists.append(arg)

        if last_size is None:
            raise ValueError("Why are you using this operator?")

        results = DslSet()
        for i in range(last_size):
            single_args = []
            for arg in args_as_lists:
                if isinstance(arg, DslSet):
                    single_args.append(arg[i])
                else:
                    single_args.append(arg)
            result = inner_operator.apply(*single_args)
            results.append(result)

        return results


# Create a set with a single element - the arg
class CreateLocalSetOperator(OperatorNode):
    def __init__(self):
        super().__init__([Any], DslSet[Any])

    def apply(self, obj: Any) -> DslSet[Any]:
        return DslSet([obj])


class AddToLocalSetOperator(OperatorNode):
    def __init__(self):
        super().__init__([DslSet[Any], Any], DslSet[Any])

    def apply(self, s: DslSet[Any], obj: Any) -> DslSet[Any]:
        s.add(obj)
        return s


class MultiplyOperator(OperatorNode):

    def __init__(self):
        super().__init__([Number, Number], Number)

    def apply(self, a: Number, b: Number) -> Number:
        return a * b


class AdditionOperator(OperatorNode):
    def __init__(self):
        super().__init__([int, int], int)

    def apply(self, a:int, b:int) -> int:
        return a + b


class SubtractionOperator(OperatorNode):
    def __init__(self):
        super().__init__([int, int], int)

    def apply(self, a:int, b:int) -> int:
        return a - b


class SumSetOperator(OperatorNode):
    def __init__(self):
        super().__init__([DslSet[int]], int)

    def apply(self, input_set: DslSet[int]) -> int:
        return sum(input_set)
