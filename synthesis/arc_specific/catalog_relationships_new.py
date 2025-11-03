from collections import defaultdict
from typing import Type, Any, List, Dict, Tuple, Callable, DefaultDict

from nodes import RelationshipNode
from relationship_primitives import NotEquals, Equals, LessThan, \
    SetContains, SetNotContains
from arc.arc_relationships import AlignedCardinal, BetweenCardinal, Touches, NotTouches, \
    ObjectContains
from synthesis.unified_value_network import UnifiedValueNetwork


# TODO reconcile with catalog_relationships.py

class RelationshipSearchStrategy:

    def __init__(self, relationship: RelationshipNode, known_port: int):
        self.relationship = relationship
        self.known_port = known_port

    # Returns list of dicts (each dict is a different set of arguments that work together) mapping port index to (type, value)
    def generate_other_argument_tuples(self, uvn: UnifiedValueNetwork, known_arg_type: Type,
                                       known_value: Any, case_index: int) -> List[Dict[int, Tuple[Type, Any]]]:
       ...


class BijectiveRelationshipStrategy(RelationshipSearchStrategy):

    # Reverse fn accepts (Type, value) and returns (Type, value)
    def __init__(self, relationship: RelationshipNode, known_port: int, reverse_fn: Callable):

        if len(relationship.input_types) != 2:
            raise Exception("Invalid port setup for bijective strategy")
        super().__init__(relationship, known_port)
        self.reverse_fn = reverse_fn
        self.return_port = 1 if known_port == 0 else 0

    def generate_other_argument_tuples(self, uvn: UnifiedValueNetwork, known_arg_type: Type,
                                       known_value: Any, case_index: int) -> List[Dict[int, Tuple[Type, Any]]]:
        return_tuple = self.reverse_fn(known_arg_type, known_value)
        return [{self.return_port: return_tuple}]


class SingleArgProposalRelationshipStrategy(RelationshipSearchStrategy):

    def __init__(self, relationship: RelationshipNode, known_port: int):
        if len(relationship.input_types) != 2:
            raise Exception("Invalid port setup for single-arg proposal strategy")
        super().__init__(relationship, known_port)
        self.return_port = 1 if known_port == 0 else 0
        self.return_type = relationship.input_types[self.return_port]

        # if self.return_type is Any:
        #     raise Exception("Generics not supported yet here...") # TODO ?

    def generate_other_argument_tuples(self, uvn: UnifiedValueNetwork, known_arg_type: Type, known_value: Any,
                                       case_index: int) -> List[Dict[int, Tuple[Type, Any]]]:
        return_list = []

        functional_return_type = self.return_type if self.return_type != Any else known_arg_type
        for search_value in uvn.get_values_by_type(functional_return_type, case_index):
            try:
                args = (known_value, search_value) if self.known_port == 0 else (search_value, known_value)
                if self.relationship.apply(*args):
                    return_list.append({self.return_port: (functional_return_type, search_value)})
            except Exception as e:
                print("Rel search exception:", e)
        return return_list

TEMP_RELATIONSHIP_LIST: List[RelationshipNode] = [
    Equals(),
    NotEquals(),
    LessThan(),
    SetContains(),
    SetNotContains(),
    AlignedCardinal(),
    BetweenCardinal(),
    Touches(),
    ObjectContains(),
    NotTouches(),
]


TEMP_RELATIONSHIP_SEARCH_STRATEGIES: DefaultDict[Type, List[RelationshipSearchStrategy]] = defaultdict(list)

def temp_add_strategy(strategy: RelationshipSearchStrategy):
    known_type = strategy.relationship.input_types[strategy.known_port]
    TEMP_RELATIONSHIP_SEARCH_STRATEGIES[known_type].append(strategy)


for rel in TEMP_RELATIONSHIP_LIST:

    if isinstance(rel, Equals):
        temp_strategy = BijectiveRelationshipStrategy(rel, 0, lambda t, v: (t, v))
        temp_add_strategy(temp_strategy)

    if isinstance(rel, NotEquals):
        temp_strategy = SingleArgProposalRelationshipStrategy(rel, 0)
        temp_add_strategy(temp_strategy)

    if isinstance(rel, Touches):
        temp_strategy = SingleArgProposalRelationshipStrategy(rel, 0)
        temp_add_strategy(temp_strategy)

    if isinstance(rel, ObjectContains):
        temp_strategy_1 = SingleArgProposalRelationshipStrategy(rel, 0)
        temp_add_strategy(temp_strategy_1)
        temp_strategy_2 = SingleArgProposalRelationshipStrategy(rel, 1)
        temp_add_strategy(temp_strategy_2)