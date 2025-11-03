from typing import Dict, Type

from nodes import Constant, InputNode, OutputNode, Node
from port_graphs import Edge
from arc.arc_operators import ApplyTwoDimensionalRelationshipOperator, \
    MeasureCentersOperator, MeasureClosestSidesOperator, TranslateIdentityOperator, ShapeSizeEnvOperator
from operator_primitives import ConstructorOperator
from relationship_primitives import Equals, NotEquals
from arc.arc_relationships import Touches, NotTouches, ObjectContains


# ApplyTwoDimensionalRelationshipOperator
# MeasureCentersOperator
# MeasureClosestSidesOperator
# TranslateIdentityOperator
# ShapeSizeEnvOperator

NODE_PRIORS: Dict[Type[Node], float] = {

    Edge: 1.0,

    Constant: 0.95,
    InputNode: 1.0,
    OutputNode: 1.0,

    ConstructorOperator: 0.9,

    ApplyTwoDimensionalRelationshipOperator: .85,
    MeasureCentersOperator: .85,
    MeasureClosestSidesOperator: .85,
    TranslateIdentityOperator: .85,
    ShapeSizeEnvOperator: .85,

    Equals: 1.0,
    NotEquals: 0.2,
    ObjectContains: 0.85,
    Touches: 0.8,
    NotTouches: 0.6
}