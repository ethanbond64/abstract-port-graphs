import unittest
from typing import Type

import numpy as np

from arc.arc_utils import read_challenges, read_solutions
from arc.arc_operators import (ApplyLinearPatternOperator,
                                   CreateRectangleOperator,
                                   DrawLineOperator, ApplyIdentityRelationshipOperator,
                                   TransformLocationRelationshipOperator, DivideIdentityOperator,
                                   SpecificLocationValueOperator, FindClosestOperator,
                                   MeasureCentersOperator,
                                   SumLocationRelationshipOperator,
                                   ApplyTwoDimensionalRelationshipOperator, CutShapeOperator,
                                   PossibleCutLocationsOperator,
                                   DivideToZonesOperator, GetZoneByIndexOperator,
                                   ZoneLocationToFullLocationOperator, GetCavityZonesOperator,
                                   GetOuterCavitiesOperator, MeasureClosestSidesOperator,
                                   MeasureIdentityDeltaOperator, GetClusterColorPatternOperator,
                                   GetClusterRadialColorPatternOperator,
                                   GetLinearShapePatternOperator, ShapeSizeEnvOperator,
                                   ProportionalEnvOperator,
                                   MoveOperator, ApproachOperator, CollapseDirectionOperator,
                                   Rotate90Operator, BinaryAndOperator, BinaryNorOperator,
                                   PersistOperator, RecolorOperator, RelocateOperator,
                                   RelocateCardinalOperator, CreateCompositeObjectOperator, SetZIndexOperator)
from arc.arc_relationships import AlignedCardinal, BetweenCardinal, Touches, NotTouches, \
    ObjectContains
from arc.arc_objects import ColorValue, Direction, ArcObject, LocationValue, CENTERED, LinearLocation, \
    LocationSpecificAttribute, LocationSchema, OneDimensionalLocationRelationship, \
    TwoDimensionalLocationRelationship, EMPTY_OBJECT, ObjectIdentity, EnvironmentShape, ArcZone, ShapeValue, \
    ArcEnvironment, ArcGraph, ArcInterpreter
from arc.arc_perception import (inner_cavities, adjacency, individual_cells,
                                    adjacency_cardinal, color_based_dividers_factory,
                                    color_based_dividers, all_cavities, adjacency_any_color)
from nodes import Constant, RecursiveProxyNode, IterativeProxyNode, \
    RuntimeOrderNode, RecursiveSearchProxyNode, SearchConditionNode, OperatorNode, SetJoin, SetSplit, InputNode, \
    OutputNode, InputSetNode, DisjointSetNode
from operator_primitives import ConstructorOperator, SetRankOperator, \
    ApplyScalarOpToSetOperator, CreateLocalSetOperator, AddToLocalSetOperator, MultiplyOperator, AdditionOperator, \
    SubtractionOperator, SumSetOperator
from relationship_primitives import Equals, NotEquals, LessThan, SetContains, \
    SetNotContains
from base_types import DslSet
from interpreter.interpreter import Interpreter


# helper node classes to avoid always passing the same type through the constructor
class ArcInputNode(InputNode):
    def __init__(self, perception_function):
        super().__init__(ArcObject, {perception_function})


class ArcOutputNode(OutputNode):
    def __init__(self):
        super().__init__(ArcObject)


# 1
def solve_007bbfb7():
    environment = ArcEnvironment([adjacency, individual_cells], 1)
    graph = ArcGraph(environment)

    # Constants
    proportion = Constant(int, 3)
    proportion_fn = Constant(int, 3)

    # Input nodes
    dot = ArcInputNode(individual_cells)
    thing = ArcInputNode(adjacency)

    # Output nodes
    output_nodes = ArcOutputNode()

    # Matrix size
    env_operator = ProportionalEnvOperator()
    graph.add_edge(environment.input_matrix_node(), env_operator, to_port=0)
    graph.add_edge(proportion, env_operator, to_port=1)
    graph.add_edge(proportion, env_operator, to_port=2)
    graph.add_edge(env_operator, environment.output_matrix_node)

    # Zones
    input_zones = DivideToZonesOperator()
    graph.add_edge(environment.input_matrix_node(), input_zones, to_port=0)
    graph.add_edge(proportion_fn, input_zones, to_port=1)
    graph.add_edge(proportion_fn, input_zones, to_port=2)

    output_zones = DivideToZonesOperator()
    graph.add_edge(environment.output_matrix_node, output_zones, to_port=0)
    graph.add_edge(proportion_fn, output_zones, to_port=1)
    graph.add_edge(proportion_fn, output_zones, to_port=2)

    # Relationships
    from_set = SetSplit(ArcZone)
    graph.add_edge(input_zones, from_set)

    zone_contains = ObjectContains()
    graph.add_edge(from_set, zone_contains, to_port=0)
    graph.add_edge(dot, zone_contains, to_port=1)

    # Operators
    output_zone = GetZoneByIndexOperator()
    graph.add_edge(output_zones, output_zone, to_port=0)
    graph.add_edge(from_set, output_zone, from_port="zone_index", to_port=1)

    final_location = ZoneLocationToFullLocationOperator()
    graph.add_edge(output_zone, final_location, to_port=0)
    graph.add_edge(thing, final_location, from_port="location", to_port=1)

    relocate = RelocateOperator()
    graph.add_edge(thing, relocate, from_port="identity", to_port=0)
    graph.add_edge(final_location, relocate, to_port=1)
    graph.add_edge(relocate, output_nodes)

    return graph


# 2
def solve_00d62c1b():
    environment = ArcEnvironment([adjacency, inner_cavities], 1)
    graph = ArcGraph(environment)

    # Constants
    yellow = Constant(ColorValue, ColorValue(5))

    # Input nodes
    outside = ArcInputNode(adjacency)
    inside = ArcInputNode(inner_cavities)

    # Output nodes
    outside_output = ArcOutputNode()
    inside_output = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    pass

    # Operators
    # persist = PersistOperator()
    graph.add_edge(outside, outside_output)
    # graph.add_edge(persist, outside_output)

    recolor = RecolorOperator()
    graph.add_edge(inside, recolor, to_port=0)
    graph.add_edge(yellow, recolor, to_port=1)
    graph.add_edge(recolor, inside_output)

    return graph


# 3
def solve_017c7c7b():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    env_shape_type = Constant(Type, EnvironmentShape)
    proportion_x = Constant(int, 1)
    proportion_y = Constant(int, 1.5)
    proportion_y_o = Constant(int, 1.5)
    vertical = Constant(Direction, Direction.DIRECTION_Y)
    red = Constant(ColorValue, ColorValue(3))

    # Input nodes
    main_object = ArcInputNode(adjacency)

    pattern = GetLinearShapePatternOperator()
    graph.add_edge(main_object, pattern, to_port=0)
    graph.add_edge(vertical, pattern, to_port=1)

    # Output nodes
    output_object = ArcOutputNode()

    # Matrix size
    multiply_x = MultiplyOperator()
    graph.add_edge(proportion_x, multiply_x, to_port=0)
    graph.add_edge(environment.input_matrix_node(), multiply_x, from_port="x_size", to_port=1)

    multiply_y = MultiplyOperator()
    graph.add_edge(proportion_y, multiply_y, to_port=0)
    graph.add_edge(environment.input_matrix_node(), multiply_y, from_port="y_size", to_port=1)

    env_shape = ConstructorOperator()
    graph.add_edge(env_shape_type, env_shape, to_port=0)
    graph.add_edge(multiply_y, env_shape, to_port=1)
    graph.add_edge(multiply_x, env_shape, to_port=2)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    pass

    # Operators
    multiply = MultiplyOperator()
    graph.add_edge(main_object, multiply, from_port="identity.shape.bounding_height", to_port=0)
    graph.add_edge(proportion_y_o, multiply, to_port=1)

    apply_linear_pattern = ApplyLinearPatternOperator()
    graph.add_edge(pattern, apply_linear_pattern, to_port=0)
    graph.add_edge(multiply, apply_linear_pattern, to_port=1)

    recolor = RecolorOperator()
    graph.add_edge(apply_linear_pattern, recolor, to_port=0)
    graph.add_edge(red, recolor, to_port=1)
    graph.add_edge(recolor, output_object)

    return graph


# 4
def solve_025d127b():
    environment = ArcEnvironment([adjacency_cardinal], 1)
    graph = ArcGraph(environment)

    # Constants
    zero = Constant(int, 0)
    one = Constant(int, 1)
    direction = Constant(Direction, Direction.DIRECTION_X)

    # Input nodes
    
    all_shapes_by_color = InputSetNode(ArcObject, perception_qualifiers={adjacency_cardinal}, group_by="identity.color")
    top_shapes = ArcInputNode(adjacency_cardinal)
    bottom_shape = ArcInputNode(adjacency_cardinal)

    # Output nodes
    top_output = ArcOutputNode()
    bottom_output = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    set_contains_1 = SetContains()
    graph.add_edge(all_shapes_by_color, set_contains_1, to_port=0)
    graph.add_edge(top_shapes, set_contains_1, to_port=1)

    set_contains_2 = SetContains()
    graph.add_edge(all_shapes_by_color, set_contains_2, to_port=0)
    graph.add_edge(bottom_shape, set_contains_2, to_port=1)

    set_rank = SetRankOperator(reverse=True)
    graph.add_edge(all_shapes_by_color, set_rank, from_port="*.location.y_location.actual_max", to_port=0)
    graph.add_edge(bottom_shape, set_rank, from_port="location.y_location.actual_max", to_port=1)

    equals = Equals()
    graph.add_edge(set_rank, equals, to_port=0)
    graph.add_edge(zero, equals, to_port=1)

    # Operators
    graph.add_edge(bottom_shape, bottom_output)

    move = MoveOperator()
    graph.add_edge(top_shapes, move, to_port=0)
    graph.add_edge(direction, move, to_port=1)
    graph.add_edge(one, move, to_port=2)
    graph.add_edge(move, top_output)

    return graph


# 5
def solve_045e512c():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    zero = Constant(int, 0)
    identity_type = Constant(Type, ObjectIdentity)
    object_type = Constant(Type, ArcObject.constructor_1)

    # Input nodes
    largest_object = ArcInputNode(adjacency)
    other_object = ArcInputNode(adjacency)
    object_set = InputSetNode(ArcObject, perception_qualifiers={adjacency})

    # Output nodes
    persistent_large_object = ArcOutputNode()
    repeated_objects = ArcOutputNode()

    # Matrix size
    pass

    # Relationships

    # NOTE Relationship operator
    set_rank = SetRankOperator(True)
    graph.add_edge(object_set, set_rank, from_port="*.identity.shape.count", to_port=0)
    graph.add_edge(largest_object, set_rank, from_port="identity.shape.count", to_port=1)

    equals = Equals()
    graph.add_edge(set_rank, equals, to_port=0)
    graph.add_edge(zero, equals, to_port=1)

    location_relationship = MeasureClosestSidesOperator()
    graph.add_edge(largest_object, location_relationship, to_port=0)
    graph.add_edge(other_object, location_relationship, to_port=1)

    # Recursive proxy
    recursive_proxy = RecursiveProxyNode()
    graph.add_edge(largest_object, recursive_proxy, to_port=0)
    graph.add_edge(repeated_objects, recursive_proxy, to_port=1)

    # Operators
    # persist
    graph.add_edge(largest_object, persistent_large_object)

    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(other_object, identity_constructor, from_port="identity.color", to_port=1)
    graph.add_edge(largest_object, identity_constructor, from_port="identity.shape", to_port=2)

    apply_location = ApplyTwoDimensionalRelationshipOperator()
    graph.add_edge(location_relationship, apply_location, to_port=0)
    graph.add_edge(recursive_proxy, apply_location, to_port=1)

    object_constructor = ConstructorOperator()
    graph.add_edge(object_type, object_constructor, to_port=0)
    graph.add_edge(identity_constructor, object_constructor, to_port=1)
    graph.add_edge(apply_location, object_constructor, to_port=2)
    graph.add_edge(object_constructor, repeated_objects)

    return graph


# 6
def solve_0520fde7():
    environment = ArcEnvironment([color_based_dividers_factory(6)], 1)
    graph = ArcGraph(environment)

    # Constants
    grey = Constant(ColorValue, ColorValue(6))
    red = Constant(ColorValue, ColorValue(3))
    center = Constant(LocationValue, CENTERED)
    identity_type = Constant(Type, ObjectIdentity)

    # Input nodes
    rhs = ArcInputNode(color_based_dividers)
    lhs = ArcInputNode(color_based_dividers)

    # Output nodes
    output = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(rhs, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    not_eq_1 = NotEquals()
    graph.add_edge(rhs, not_eq_1, from_port="identity.color", to_port=0)
    graph.add_edge(grey, not_eq_1, to_port=1)

    not_eq_2 = NotEquals()
    graph.add_edge(lhs, not_eq_2, from_port="identity.color", to_port=0)
    graph.add_edge(grey, not_eq_2, to_port=1)

    # Operators
    and_op = BinaryAndOperator()
    graph.add_edge(rhs, and_op, from_port="identity", to_port=0)
    graph.add_edge(lhs, and_op, from_port="identity", to_port=1)

    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(red, identity_constructor, to_port=1)
    graph.add_edge(and_op, identity_constructor, from_port="shape", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(identity_constructor, relocate, to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, output)

    return graph


# 7
def solve_05269061():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    object_sets = Constant(DslSet, get_05269061_constant_set())

    # Input node
    object_node = ArcInputNode(adjacency)

    # Output nodes
    original_object = ArcOutputNode()
    related_objects = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    related_set = SetSplit(DslSet)
    graph.add_edge(object_sets, related_set)

    related_set_member = SetSplit(ArcObject)
    graph.add_edge(related_set, related_set_member)

    other_set_member = SetSplit(ArcObject)
    graph.add_edge(related_set, other_set_member)

    equals_1 = Equals()
    graph.add_edge(object_node, equals_1, from_port="identity.shape", to_port=0)
    graph.add_edge(related_set_member, equals_1, from_port="identity.shape", to_port=1)

    equals_2 = Equals()
    graph.add_edge(object_node, equals_2, from_port="location", to_port=0)
    graph.add_edge(related_set_member, equals_2, from_port="location", to_port=1)

    # Operators
    graph.add_edge(object_node, original_object)

    recolor = RecolorOperator()
    graph.add_edge(other_set_member, recolor, to_port=0)
    graph.add_edge(object_node, recolor, from_port="identity.color", to_port=1)
    graph.add_edge(recolor, related_objects)

    return graph


def get_05269061_constant_set():

    set_1 = DslSet({
        ArcObject([[1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1]])
    })

    set_2 = DslSet({
        ArcObject([[0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0]])
    })

    set_3 = DslSet({
        ArcObject([[0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0]]),
        ArcObject([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 1, 0]])
    })

    return DslSet([set_1, set_2, set_3])


# 8
def solve_05f2a901():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    red = Constant(ColorValue, ColorValue(3))
    blue = Constant(ColorValue, ColorValue(9))

    # Input nodes
    
    red_object = ArcInputNode(adjacency)
    blue_object = ArcInputNode(adjacency)

    # Output nodes
    red_output = ArcOutputNode()
    blue_output = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_1 = Equals()
    graph.add_edge(red_object, equals_1, from_port="identity.color", to_port=0)
    graph.add_edge(red, equals_1, to_port=1)
    equals_2 = Equals()
    graph.add_edge(blue_object, equals_2, from_port="identity.color", to_port=0)
    graph.add_edge(blue, equals_2, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(blue_object, persist)
    graph.add_edge(persist, blue_output)

    approach = ApproachOperator()
    graph.add_edge(red_object, approach, to_port=0)
    graph.add_edge(blue_object, approach, to_port=1)
    graph.add_edge(approach, red_output)

    return graph


# 9
def solve_06df4c85():
    environment = ArcEnvironment([adjacency, all_cavities], 1)
    graph = ArcGraph(environment)

    # Constants
    pass

    # Input nodes
    
    color_other = ArcInputNode(adjacency)
    color_node_1 = ArcInputNode(adjacency)
    color_node_2 = ArcInputNode(adjacency)
    cavity_node = ArcInputNode(all_cavities)

    # Output nodes
    color_other_output = ArcOutputNode()
    new_output = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(color_node_1, equals, from_port="identity.color", to_port=0)
    graph.add_edge(color_node_2, equals, from_port="identity.color", to_port=1)

    aligned_colors = AlignedCardinal()
    graph.add_edge(color_node_1, aligned_colors, to_port=0)
    graph.add_edge(color_node_2, aligned_colors, to_port=1)

    aligned_1 = AlignedCardinal()
    graph.add_edge(color_node_1, aligned_1, to_port=0)
    graph.add_edge(cavity_node, aligned_1, to_port=1)

    aligned_2 = AlignedCardinal()
    graph.add_edge(color_node_2, aligned_2, to_port=0)
    graph.add_edge(cavity_node, aligned_2, to_port=1)

    between = BetweenCardinal()
    graph.add_edge(cavity_node, between, to_port=0)
    graph.add_edge(color_node_1, between, to_port=1)
    graph.add_edge(color_node_2, between, to_port=2)

    # Operators
    persist_operator_0 = PersistOperator()
    graph.add_edge(color_other, persist_operator_0)
    graph.add_edge(persist_operator_0, color_other_output)

    recolor_operator = RecolorOperator()
    graph.add_edge(cavity_node, recolor_operator, to_port=0)
    graph.add_edge(color_node_1, recolor_operator, from_port="identity.color", to_port=1)
    graph.add_edge(recolor_operator, new_output)

    return graph


# 10
def solve_08ed6ac7():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    zero = Constant(int, 0)
    one = Constant(int, 1)
    two = Constant(int, 2)
    three = Constant(int, 3)
    yellow = Constant(ColorValue, ColorValue(5))
    green = Constant(ColorValue, ColorValue(4))
    red = Constant(ColorValue, ColorValue(3))
    blue = Constant(ColorValue, ColorValue(2))

    # Input nodes
    
    gray_object = ArcInputNode(adjacency)

    # Output nodes
    colored_object = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    gray_object_set = SetJoin()
    graph.add_edge(gray_object, gray_object_set)

    # Operators
    set_rank = SetRankOperator()
    graph.add_edge(gray_object_set, set_rank, from_port="*.identity.shape.bounding_height", to_port=0)
    graph.add_edge(gray_object, set_rank, from_port="identity.shape.bounding_height", to_port=1)

    equals = Equals()
    graph.add_edge(set_rank, equals, to_port=0)
    graph.add_generic_edge([zero, one, two, three], equals, to_port=0)

    recolor = RecolorOperator()
    graph.add_edge(gray_object, recolor, to_port=0)
    graph.add_generic_edge([yellow, green, red, blue], recolor, to_port=1)
    graph.add_edge(recolor, colored_object)

    return graph


# 11
def solve_x_09629e4f():
    environment = ArcEnvironment([adjacency_cardinal], 1)
    graph = ArcGraph(environment)

    # Constants
    four = Constant(int, 4)
    three = Constant(int, 3)
    grey = Constant(ColorValue, ColorValue(6))
    box = Constant(ArcObject, ArcObject.create_zone_object([[1, 1, 1],
                                                            [1, 1, 1],
                                                            [1, 1, 1]]))
    identity_type = Constant(Type, ObjectIdentity)

    # Input nodes
    dot = ArcInputNode(adjacency_cardinal)
    divider = ArcInputNode(adjacency_cardinal)

    # Output nodes
    output_divider = ArcOutputNode()
    output_box = ArcOutputNode()

    # Relationships
    cavity_zones = GetCavityZonesOperator()
    graph.add_edge(divider, cavity_zones, to_port=0)

    cavity_zone = SetSplit(ArcZone)
    graph.add_edge(cavity_zones, cavity_zone)

    zone_contains = ObjectContains()
    graph.add_edge(cavity_zone, zone_contains, to_port=0)
    graph.add_edge(dot, zone_contains, to_port=1)


    # Get the divider (by shape or color)
    # Get each dot
    # get the dividers cavity zones
    # Set join by zone?
    # Set of sets
    # Current set rank vs other sets
    # Get largest
    # get indv cells zones of current zone
    # Compare zone index to outer zone indexes
    # Create 3x3 square with dot color and place it in outer zone...


    nine_square_set = InputSetNode(ArcObject, perception_qualifiers={color_based_dividers},
                                   group_by="identity.shape.bounding_area")
    cell_set = InputSetNode(ArcObject, perception_qualifiers={individual_cells}, group_by="parent_id")

    # Output nodes
    output_divider = ArcOutputNode()
    output_nine_square = ArcOutputNode()

    # Matrix size
    # pass
    #
    # # Relationships
    # equals_1 = Equals()
    # graph.add_edge(grey, equals_1, to_port=0)
    # graph.add_edge(divider, equals_1, from_port="identity.color", to_port=1)
    #
    # equals_2 = Equals()
    # graph.add_edge(nine_square_set, equals_2, from_port="size", to_port=0)
    # graph.add_edge(nine, equals_2, to_port=1)
    #
    # equals_3 = Equals()
    # graph.add_edge(cell_set, equals_3, from_port="size", to_port=0)
    # graph.add_edge(four, equals_3, to_port=1)
    #
    # equals_4 = Equals()
    # graph.add_edge(cell, equals_4, from_port="zone_info", to_port=0)
    # graph.add_edge(nine_square, equals_4, from_port="zone_info", to_port=1)
    #
    # member_of_set_1 = SetJoin()
    # graph.add_edge(nine_square_set, nine_square)
    #
    # member_of_set_2 = SetJoin()
    # graph.add_edge(cell_set, cell)
    #
    # # Operators
    # persist = PersistOperator()
    # graph.add_edge(divider, persist)
    # graph.add_edge(persist, output_divider)
    #
    # identity_constructor = ConstructorOperator()
    # graph.add_edge(identity_type, identity_constructor)
    # graph.add_edge(cell, identity_constructor, from_port="identity.color", to_port=1)
    # graph.add_edge(box, identity_constructor, from_port="identity.shape", to_port=2)
    #
    # relocate = RelocateOperator()
    # graph.add_edge(identity_constructor, relocate, to_port=0)
    # graph.add_edge(nine_square, relocate, from_port="location", to_port=1)
    # graph.add_edge(relocate, output_nine_square)
    #
    # return graph


# 12
def solve_0962bcdd():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    dot = Constant(ArcObject, ArcObject([[1]]))
    mini_cross = Constant(ArcObject, ArcObject([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]]))
    generic_filter_constant = [dot, mini_cross]
    diagonal_cross = Constant(ArcObject, ArcObject([[1, 0, 0, 0, 1],
                                                    [0, 1, 0, 1, 0],
                                                    [0, 0, 1, 0, 0],
                                                    [0, 1, 0, 1, 0],
                                                    [1, 0, 0, 0, 1]]))
    vertical_cross = Constant(ArcObject, ArcObject([[0, 0, 1, 0, 0],
                                                    [0, 0, 1, 0, 0],
                                                    [1, 1, 0, 1, 1],
                                                    [0, 0, 1, 0, 0],
                                                    [0, 0, 1, 0, 0]]))
    generic_operator_constant = [diagonal_cross, vertical_cross]

    # Input nodes
    
    input_object = ArcInputNode(adjacency)

    # Output nodes
    output_object = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(input_object, equals, from_port="identity.shape", to_port=0)
    graph.add_generic_edge(generic_filter_constant, equals, from_port="identity.shape", to_port=1)

    # Operators
    recolor_diagonal = RecolorOperator()
    graph.add_generic_edge(generic_operator_constant, recolor_diagonal, to_port=0)
    graph.add_edge(input_object, recolor_diagonal, from_port="identity.color", to_port=1)

    relocate_diagonal = RelocateOperator()
    graph.add_edge(recolor_diagonal, relocate_diagonal, from_port="identity", to_port=0)
    graph.add_edge(input_object, relocate_diagonal, from_port="location", to_port=1)
    graph.add_edge(relocate_diagonal, output_object)

    return graph


# 13
def solve_0a938d79():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    first_object = ArcInputNode(adjacency)
    second_object = ArcInputNode(adjacency)

    # Output nodes
    first_output_cluster = ArcOutputNode()
    output_cluster = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_generic_edge(first_object, equals, from_port=["location.y_location.boundary", "location.x_location.boundary"], to_port=0) # generic port
    graph.add_edge(one, equals, to_port=1)

    less_than = LessThan()
    graph.add_generic_edge(first_object, less_than, from_port=["location.x_location.center", "location.y_location.center"], to_port=0) # generic port
    graph.add_generic_edge(second_object, less_than, from_port=["location.x_location.center", "location.y_location.center"], to_port=1) # generic port

    create_rectangle_first = CreateRectangleOperator()
    graph.add_edge(first_object, create_rectangle_first, from_port="identity.color", to_port=0)
    graph.add_generic_edge(first_object, create_rectangle_first, from_port=["identity.shape.bounding_width", "identity.shape.bounding_height"], to_port=[1, 2]) # generic edge order and port...
    graph.add_generic_edge(environment.input_matrix_node(), create_rectangle_first, from_port=["y_size", "x_size"], to_port=[2, 1]) # generic edge order and port...

    locate_rectangle_first = RelocateCardinalOperator()
    graph.add_edge(create_rectangle_first, locate_rectangle_first, to_port=0)
    graph.add_generic_edge(first_object, locate_rectangle_first, from_port=["location.x_location", "location.y_location"], to_port=[1, 2]) # generic edge order and port...
    graph.add_generic_edge(center, locate_rectangle_first, from_port=["y_location", "x_location"], to_port=[2, 1]) # generic edge order and port...

    create_rectangle_second = CreateRectangleOperator()
    graph.add_edge(second_object, create_rectangle_second, from_port="identity.color", to_port=0)
    graph.add_generic_edge(second_object, create_rectangle_second, from_port=["identity.shape.bounding_width", "identity.shape.bounding_height"], to_port=[1, 2])  # generic edge order and port...
    graph.add_generic_edge(environment.input_matrix_node(), create_rectangle_second, from_port=["y_size", "x_size"], to_port=[2, 1])  # generic edge order and port...

    locate_rectangle_second = RelocateCardinalOperator()
    graph.add_edge(create_rectangle_second, locate_rectangle_second, to_port=0)
    graph.add_generic_edge(second_object, locate_rectangle_second, from_port=["location.x_location", "location.y_location"], to_port=[1, 2]) # generic edge order and port...
    graph.add_generic_edge(center, locate_rectangle_second, from_port=["y_location", "x_location"], to_port=[2, 1]) # generic edge order and port...

    abstract_location_relationship = MeasureClosestSidesOperator()
    graph.add_edge(locate_rectangle_first, abstract_location_relationship, to_port=0)
    graph.add_edge(locate_rectangle_second, abstract_location_relationship, to_port=1)

    set_member_first = CreateLocalSetOperator()
    graph.add_edge(locate_rectangle_first, set_member_first)

    cluster_set = AddToLocalSetOperator()
    graph.add_edge(set_member_first, cluster_set, to_port=0)
    graph.add_edge(locate_rectangle_second, cluster_set, to_port=1)

    set_to_cluster = CreateCompositeObjectOperator()
    graph.add_edge(cluster_set, set_to_cluster)
    graph.add_edge(set_to_cluster, first_output_cluster)

    # Recursive proxy
    recursive_proxy = RecursiveProxyNode()
    graph.add_edge(set_to_cluster, recursive_proxy, to_port=0)
    graph.add_edge(output_cluster, recursive_proxy, to_port=1)

    apply_location = ApplyTwoDimensionalRelationshipOperator()
    graph.add_edge(abstract_location_relationship, apply_location, to_port=0)
    graph.add_edge(recursive_proxy, apply_location, to_port=1)

    relocate = RelocateOperator()
    graph.add_edge(set_to_cluster, relocate, from_port="identity", to_port=0)
    graph.add_edge(apply_location, relocate, to_port=1)
    graph.add_edge(relocate, output_cluster)

    return graph


# 14
def solve_0b148d64():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    zero = Constant(int, 0)
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    color_set = InputSetNode(ArcObject, perception_qualifiers={adjacency}, group_by="identity.color")

    # Output nodes
    output_cluster = ArcOutputNode()

    set_to_cluster = CreateCompositeObjectOperator()
    graph.add_edge(color_set, set_to_cluster)

    # Relationships
    set_member = SetJoin()
    graph.add_edge(set_to_cluster, set_member)

    set_rank = SetRankOperator()
    graph.add_edge(set_member, set_rank, "*.identity.shape.count", to_port=0)
    graph.add_edge(set_to_cluster, set_rank, from_port="identity.shape.count", to_port=1)

    equals = Equals()
    graph.add_edge(set_rank, equals, to_port=0)
    graph.add_edge(zero, equals, to_port=1)

    # Operators
    relocate = RelocateOperator()
    graph.add_edge(set_to_cluster, relocate, from_port="identity", to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, output_cluster)

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(output_cluster, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    return graph


# 15
def solve_0ca9ddb6():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    red = Constant(ColorValue, ColorValue(3))
    blue = Constant(ColorValue, ColorValue(2))
    yellow = Constant(ArcObject, ArcObject([[5, 0, 5],
                                            [0, 0, 0],
                                            [5, 0, 5]]))
    orange = Constant(ArcObject, ArcObject([[0, 8, 0],
                                            [8, 0, 8],
                                            [0, 8, 0]]))

    # Input nodes
    
    red_dot = ArcInputNode(adjacency)
    blue_dot = ArcInputNode(adjacency)
    other_dot = ArcInputNode(adjacency)

    # Output nodes
    yellow_output = ArcOutputNode()
    orange_output = ArcOutputNode()
    red_output = ArcOutputNode()
    blue_output = ArcOutputNode()
    other_output = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_1 = Equals()
    graph.add_edge(red_dot, equals_1, from_port="identity.color", to_port=0)
    graph.add_edge(red, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(blue_dot, equals_2, from_port="identity.color", to_port=0)
    graph.add_edge(blue, equals_2, to_port=1)

    # Operators
    persist_1 = PersistOperator()
    graph.add_edge(red_dot, persist_1)
    graph.add_edge(persist_1, red_output)

    persist_2 = PersistOperator()
    graph.add_edge(blue_dot, persist_2)
    graph.add_edge(persist_2, blue_output)

    persist_3 = PersistOperator()
    graph.add_edge(other_dot, persist_3)
    graph.add_edge(persist_3, other_output)

    relocate_yellow = RelocateOperator()
    graph.add_edge(yellow, relocate_yellow, "identity", to_port=0)
    graph.add_edge(red_dot, relocate_yellow, from_port="location", to_port=1)
    graph.add_edge(relocate_yellow, yellow_output)

    relocate_orange = RelocateOperator()
    graph.add_edge(orange, relocate_orange, "identity", to_port=0)
    graph.add_edge(blue_dot, relocate_orange, from_port="location", to_port=1)
    graph.add_edge(relocate_orange, orange_output)

    return graph


# 16
def solve_0d3d703e():
    environment = ArcEnvironment([adjacency], None)
    graph = ArcGraph(environment)

    # Constants
    green = Constant(ColorValue, ColorValue(4))
    green_out = Constant(ColorValue, ColorValue(4))
    yellow = Constant(ColorValue, ColorValue(5))
    yellow_out = Constant(ColorValue, ColorValue(5))

    blue = Constant(ColorValue, ColorValue(2))
    blue_out = Constant(ColorValue, ColorValue(2))
    grey = Constant(ColorValue, ColorValue(6))
    grey_out = Constant(ColorValue, ColorValue(6))

    red = Constant(ColorValue, ColorValue(3))
    red_out = Constant(ColorValue, ColorValue(3))
    pink = Constant(ColorValue, ColorValue(7))
    pink_out = Constant(ColorValue, ColorValue(7))

    teal = Constant(ColorValue, ColorValue(9))
    teal_out = Constant(ColorValue, ColorValue(9))
    maroon = Constant(ColorValue, ColorValue(10))
    maroon_out = Constant(ColorValue, ColorValue(10))

    # Input nodes
    
    object_node = ArcInputNode(adjacency)

    # Output nodes
    output_object = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(object_node, equals, from_port="identity.color", to_port=0)
    graph.add_generic_edge([
        green, yellow,
        blue, grey,
        red, pink,
        teal, maroon], equals, to_port=1)

    # Operators
    recolor = RecolorOperator()
    graph.add_edge(object_node, recolor, to_port=0)
    graph.add_generic_edge([
        yellow_out, green_out,
        grey_out, blue_out,
        pink_out, red_out,
        maroon_out, teal_out
    ], recolor, to_port=1)
    graph.add_edge(recolor, output_object)

    return graph


# 17
def solve_0dfd9992():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    all_objects = InputSetNode(ArcObject, perception_qualifiers={adjacency})

    # Output nodes
    main_output_node = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(main_output_node, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships

    cluster_node = CreateCompositeObjectOperator()
    graph.add_edge(all_objects, cluster_node)

    pattern = GetClusterColorPatternOperator()
    graph.add_edge(cluster_node, pattern)

    # Operators
    create_rectangle = CreateRectangleOperator()
    graph.add_edge(pattern, create_rectangle, to_port=0)
    graph.add_edge(environment.input_matrix_node(), create_rectangle, from_port="x_size", to_port=1)
    graph.add_edge(environment.input_matrix_node(), create_rectangle, from_port="y_size", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(create_rectangle, relocate, to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, main_output_node)

    return graph


# 18
def solve_x_0e206a2e():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    dot = Constant(ArcObject, ArcObject([[1]]))

    # Input objects
    
    thing = ArcInputNode(adjacency)
    start_1 = ArcInputNode(adjacency)
    start_2 = ArcInputNode(adjacency)
    start_3 = ArcInputNode(adjacency)
    end_1 = ArcInputNode(adjacency)
    end_2 = ArcInputNode(adjacency)
    end_3 = ArcInputNode(adjacency)

    # Output objects
    output_thing = ArcOutputNode()
    end_out_1 = ArcOutputNode()
    end_out_2 = ArcOutputNode()
    end_out_3 = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_shape_1 = Equals()
    graph.add_edge(start_1, equals_shape_1, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, equals_shape_1, from_port="identity.shape", to_port=1)

    equals_shape_2 = Equals()
    graph.add_edge(start_2, equals_shape_2, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, equals_shape_2, from_port="identity.shape", to_port=1)

    equals_shape_3 = Equals()
    graph.add_edge(start_3, equals_shape_3, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, equals_shape_3, from_port="identity.shape", to_port=1)

    equals_shape_4 = Equals()
    graph.add_edge(end_1, equals_shape_4, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, equals_shape_4, from_port="identity.shape", to_port=1)

    equals_shape_5 = Equals()
    graph.add_edge(end_2, equals_shape_5, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, equals_shape_5, from_port="identity.shape", to_port=1)

    equals_shape_6 = Equals()
    graph.add_edge(end_3, equals_shape_6, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, equals_shape_6, from_port="identity.shape", to_port=1)

    equals_color_1 = Equals()
    graph.add_edge(start_1, equals_color_1, from_port="identity.color", to_port=0)
    graph.add_edge(end_1, equals_color_1, from_port="identity.color", to_port=1)

    equals_color_2 = Equals()
    graph.add_edge(start_2, equals_color_2, from_port="identity.color", to_port=0)
    graph.add_edge(end_2, equals_color_2, from_port="identity.color", to_port=1)

    equals_color_3 = Equals()
    graph.add_edge(start_3, equals_color_3, from_port="identity.color", to_port=0)
    graph.add_edge(end_3, equals_color_3, from_port="identity.color", to_port=1)

    not_equals = NotEquals()
    graph.add_edge(thing, not_equals, from_port="identity.shape", to_port=0)
    graph.add_edge(dot, not_equals, from_port="identity.shape", to_port=1)

    touches_1 = Touches()
    graph.add_edge(start_1, touches_1, to_port=0)
    graph.add_edge(thing, touches_1, to_port=1)

    touches_2 = Touches()
    graph.add_edge(start_2, touches_2, to_port=0)
    graph.add_edge(thing, touches_2, to_port=1)

    touches_3 = Touches()
    graph.add_edge(start_3, touches_3, to_port=0)
    graph.add_edge(thing, touches_3, to_port=1)

    not_touches_1 = NotTouches()
    graph.add_edge(end_1, not_touches_1, to_port=0)
    graph.add_edge(thing, not_touches_1, to_port=1)

    not_touches_2 = NotTouches()
    graph.add_edge(end_1, not_touches_2, to_port=0)
    graph.add_edge(thing, not_touches_2, to_port=1)

    not_touches_3 = NotTouches()
    graph.add_edge(end_3, not_touches_3, to_port=0)
    graph.add_edge(thing, not_touches_3, to_port=1)

    create_start_set = CreateLocalSetOperator()
    graph.add_edge(start_1, create_start_set)

    add_to_start_set_1 = AddToLocalSetOperator()
    graph.add_edge(create_start_set, add_to_start_set_1, to_port=0)
    graph.add_edge(start_2, add_to_start_set_1, to_port=1)

    add_to_start_set_2 = AddToLocalSetOperator()
    graph.add_edge(add_to_start_set_1, add_to_start_set_2, to_port=0)
    graph.add_edge(start_3, add_to_start_set_2, to_port=1)

    # Create two cluster objects from and end object from start and end objects respectively. Normalized shapes must match
    cluster_start = CreateCompositeObjectOperator()
    graph.add_edge(add_to_start_set_2, cluster_start, to_port=0)

    create_end_set = CreateLocalSetOperator()
    graph.add_edge(end_1, create_end_set, to_port=0)

    add_to_end_set_1 = AddToLocalSetOperator()
    graph.add_edge(create_end_set, add_to_end_set_1, to_port=0)
    graph.add_edge(end_2, add_to_end_set_1, to_port=1)

    add_to_end_set_2 = AddToLocalSetOperator()
    graph.add_edge(add_to_end_set_1, add_to_end_set_2, to_port=0)
    graph.add_edge(end_3, add_to_end_set_2, to_port=1)

    cluster_end = CreateCompositeObjectOperator() # TODO this isn't showing up in the input subgraph because it is an operator that constributes to the output. Need to make an exception in subgraph separation... if it contributes to a relationship
    graph.add_edge(add_to_end_set_2, cluster_end, to_port=0)

    equals_norm_shape = Equals()
    graph.add_edge(cluster_start, equals_norm_shape, from_port="identity.shape.normalized_shape", to_port=0)
    graph.add_edge(cluster_end, equals_norm_shape, from_port="identity.shape.normalized_shape", to_port=1)

    abstract_identity_relationship = MeasureIdentityDeltaOperator()
    graph.add_edge(cluster_start, abstract_identity_relationship, from_port="identity", to_port=0)
    graph.add_edge(cluster_end, abstract_identity_relationship, from_port="identity", to_port=1)

    abstract_location_relationship = MeasureCentersOperator()
    graph.add_edge(cluster_start, abstract_location_relationship, to_port=0)
    graph.add_edge(thing, abstract_location_relationship, to_port=1)

    # TODO transform location using identity relationship
    transform_location_relationship = TransformLocationRelationshipOperator()
    graph.add_edge(abstract_identity_relationship, transform_location_relationship, to_port=0)
    graph.add_edge(abstract_location_relationship, transform_location_relationship, to_port=1)

    # Operators
    persist_1 = PersistOperator()
    graph.add_edge(end_1, persist_1)
    graph.add_edge(persist_1, end_out_1)

    persist_2 = PersistOperator()
    graph.add_edge(end_2, persist_2)
    graph.add_edge(persist_2, end_out_2)

    persist_3 = PersistOperator()
    graph.add_edge(end_3, persist_3)
    graph.add_edge(persist_3, end_out_3)

    apply_identity_relationship = ApplyIdentityRelationshipOperator()
    graph.add_edge(abstract_identity_relationship, apply_identity_relationship, to_port=0)
    graph.add_edge(thing, apply_identity_relationship, from_port="identity", to_port=1)

    apply_location_relationship = ApplyTwoDimensionalRelationshipOperator() # TODO apply 2d...
    graph.add_edge(transform_location_relationship, apply_location_relationship, to_port=0)
    graph.add_edge(cluster_end, apply_location_relationship, to_port=1)

    relocate = RelocateOperator()
    graph.add_edge(apply_identity_relationship, relocate, to_port=0)
    graph.add_edge(apply_location_relationship, relocate, to_port=1)
    graph.add_edge(relocate, output_thing)

    return graph


# 19
def solve_10fcaaa3():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    two = Constant(int, 2)
    zero_1 = Constant(int, 0)
    zero_2 = Constant(int, 0)
    zero_3 = Constant(int, 0)
    zero_4 = Constant(int, 0)
    one_1 = Constant(int, 1)
    one_2 = Constant(int, 1)
    one_3 = Constant(int, 1)
    one_4 = Constant(int, 1)
    x_1 = Constant(Direction, Direction.DIRECTION_X)
    y_1 = Constant(Direction, Direction.DIRECTION_Y)
    aux_1 = Constant(ArcObject, ArcObject([[9, 0, 9],
                                           [0, 0, 0],
                                           [9, 0, 9]]))
    z_index_1 = Constant(int, 1)

    # Inputs
    dot = ArcInputNode(adjacency)

    # Outputs
    out_dot = ArcOutputNode()
    out_aux = ArcOutputNode()

    # Env shape
    env_shape_operator = ProportionalEnvOperator()
    graph.add_edge(environment.input_matrix_node(), env_shape_operator, to_port=0)
    graph.add_edge(two, env_shape_operator, to_port=1)
    graph.add_edge(two, env_shape_operator, to_port=2)
    graph.add_edge(env_shape_operator, environment.output_matrix_node)

    # Relationships
    pass

    # Operators
    multiply_x = MultiplyOperator()
    graph.add_edge(environment.input_matrix_node(), multiply_x, from_port="x_size", to_port=0)
    graph.add_generic_edge([zero_1, one_1, zero_2, one_2], multiply_x, to_port=1)

    multiply_y = MultiplyOperator()
    graph.add_edge(environment.input_matrix_node(), multiply_y, from_port="y_size", to_port=0)
    graph.add_generic_edge([zero_3, zero_4, one_3, one_4], multiply_y, to_port=1)

    move_x_1 = MoveOperator()
    graph.add_edge(dot, move_x_1, to_port=0)
    graph.add_edge(x_1, move_x_1, to_port=1)
    graph.add_edge(multiply_x, move_x_1, to_port=2)

    move_y = MoveOperator()
    graph.add_edge(move_x_1, move_y, to_port=0)
    graph.add_edge(y_1, move_y, to_port=1)
    graph.add_edge(multiply_y, move_y, to_port=2)

    set_z_index = SetZIndexOperator()
    graph.add_edge(move_y, set_z_index, to_port=0)
    graph.add_edge(z_index_1, set_z_index, to_port=1)
    graph.add_edge(set_z_index, out_dot)

    relocate = RelocateOperator()
    graph.add_edge(aux_1, relocate, from_port="identity", to_port=0)
    graph.add_edge(move_y, relocate, from_port="location", to_port=1)
    graph.add_edge(relocate, out_aux)

    return graph


# 20
def solve_11852cab():
    environment = ArcEnvironment([adjacency_any_color], 1)
    graph = ArcGraph(environment)

    # Constants
    out_shape = Constant(ArcObject, ArcObject([[2, 0, 2, 0, 2],
                                               [0, 2, 0, 2, 0],
                                               [2, 0, 2, 0, 2],
                                               [0, 2, 0, 2, 0],
                                               [2, 0, 2, 0, 2],]))
    identity_type = Constant(Type, ObjectIdentity)

    # Input objects
    things = InputSetNode(ArcObject, perception_qualifiers={adjacency_any_color})

    # Output objects
    output_thing = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    set_to_cluster = CreateCompositeObjectOperator()
    graph.add_edge(things, set_to_cluster)

    color_pattern = GetClusterRadialColorPatternOperator()
    graph.add_edge(set_to_cluster, color_pattern)

    # Operators
    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(color_pattern, identity_constructor, to_port=1)
    graph.add_edge(out_shape, identity_constructor, from_port="identity.shape", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(identity_constructor, relocate, to_port=0)
    graph.add_edge(set_to_cluster, relocate, from_port="location", to_port=1)
    graph.add_edge(relocate, output_thing)

    return graph


# 21
def solve_1190e5a7():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    input_node_1 = ArcInputNode(adjacency)
    input_node_2 = ArcInputNode(adjacency)

    # Output nodes
    rectangle_output = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(rectangle_output, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    equals_1 = Equals()
    graph.add_edge(input_node_1, equals_1, from_port="location.y_location.min_boundary", to_port=0)
    graph.add_edge(one, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(input_node_2, equals_2, from_port="location.x_location.min_boundary", to_port=0)
    graph.add_edge(one, equals_2, to_port=1)

    equals_3 = Equals()
    graph.add_edge(input_node_1, equals_3, from_port="identity.color.value", to_port=0)
    graph.add_edge(input_node_2, equals_3, from_port="identity.color.value", to_port=1)

    input_node_set_1 = SetJoin()
    graph.add_edge(input_node_1, input_node_set_1)

    input_node_set_2 = SetJoin()
    graph.add_edge(input_node_2, input_node_set_2)

    # Operators
    create_rectangle = CreateRectangleOperator()
    graph.add_edge(input_node_1, create_rectangle, from_port="identity.color", to_port=0)
    graph.add_edge(input_node_set_1, create_rectangle, from_port="size", to_port=1)
    graph.add_edge(input_node_set_2, create_rectangle, from_port="size", to_port=2)

    relocate_rectangle = RelocateOperator()
    graph.add_edge(create_rectangle, relocate_rectangle, to_port=0)
    graph.add_edge(center, relocate_rectangle, to_port=1)
    graph.add_edge(relocate_rectangle, rectangle_output)

    return graph


# 22
def solve_137eaa0f():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    grey = Constant(ColorValue, ColorValue(6))
    grey_center = Constant(ArcObject, ArcObject([[0, 0, 0],
                                                 [0, 6, 0],
                                                 [0, 0, 0]]))
    matrix = Constant(ArcObject, ArcObject([[1, 1, 1],
                                             [1, 1, 1],
                                             [1, 1, 1]]))

    # Input nodes
    grey_dot = ArcInputNode(adjacency)
    color_object = ArcInputNode(adjacency)

    # Output nodes
    output_center = ArcOutputNode()
    output_color = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(matrix, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    equals = Equals()
    graph.add_edge(grey_dot, equals, from_port="identity.color", to_port=0)
    graph.add_edge(grey, equals, to_port=1)

    not_equals = NotEquals()
    graph.add_edge(color_object, not_equals, from_port="identity.color", to_port=0)
    graph.add_edge(grey, not_equals, to_port=1)

    touches = Touches()
    graph.add_edge(grey_dot, touches, to_port=0)
    graph.add_edge(color_object, touches, to_port=1)

    sides_distance = MeasureClosestSidesOperator()
    graph.add_edge(grey_dot, sides_distance, to_port=0)
    graph.add_edge(color_object, sides_distance, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(grey_center, persist)
    graph.add_edge(persist, output_center)

    created_location = ApplyTwoDimensionalRelationshipOperator()
    graph.add_edge(sides_distance, created_location, to_port=0)
    graph.add_edge(grey_center, created_location, to_port=1)

    relocate = RelocateOperator()
    graph.add_edge(color_object, relocate, from_port="identity", to_port=0)
    graph.add_edge(created_location, relocate, to_port=1)
    graph.add_edge(relocate, output_color)

    return graph


# 23
def solve_TOOSLOW_150deff5():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    zero = Constant(int, 0)
    empty = Constant(ArcObject, EMPTY_OBJECT)
    square = Constant(ArcObject, ArcObject([[9, 9],
                                            [9, 9]]))
    line_horizontal = Constant(ArcObject, ArcObject([[3, 3, 3]]))
    line_vertical = Constant(ArcObject, ArcObject([[3],
                                                   [3],
                                                   [3]]))
    relocate_op_var = Constant(OperatorNode, RelocateOperator())

    # Inputs
    
    grey_thing = ArcInputNode(adjacency)

    # Outputs
    out_objects = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    pass

    # Operators
    functional_grey_thing = RecursiveProxyNode()
    graph.add_edge(grey_thing, functional_grey_thing, to_port=0)
    # NOTE second arg comes from `cut_shape` below

    ### SEARCH CONDITION

    node_to_place = RecursiveSearchProxyNode()
    graph.add_edge(empty, node_to_place, to_port=0)
    # NOTE second arg comes from `create_next_objects` below

    cut_shape = CutShapeOperator()
    graph.add_edge(functional_grey_thing, cut_shape, to_port=0)
    graph.add_edge(node_to_place, cut_shape, from_port="identity.shape", to_port=1)
    graph.add_edge(node_to_place, cut_shape, from_port="location", to_port=2)
    graph.add_edge(cut_shape, functional_grey_thing, to_port=1)

    search_condition = SearchConditionNode(Equals())
    graph.add_edge(cut_shape, search_condition, from_port="identity.shape.count", to_port=0)
    graph.add_edge(zero, search_condition, to_port=1)

    potential_locations = PossibleCutLocationsOperator()
    graph.add_edge(cut_shape, potential_locations, to_port=0)
    graph.add_generic_edge([square, line_horizontal, line_vertical], potential_locations, from_port="identity", to_port=1)

    create_next_objects = ApplyScalarOpToSetOperator()
    graph.add_edge(relocate_op_var, create_next_objects, to_port=0)
    graph.add_generic_edge([square, line_horizontal, line_vertical], create_next_objects, from_port="identity", to_port=1)
    graph.add_edge(potential_locations, create_next_objects, to_port=2)
    graph.add_edge(create_next_objects, node_to_place, to_port=3) # NOTE RECURSIVE SEARCH OUTPUT

    persist = PersistOperator()
    graph.add_edge(node_to_place, persist)
    graph.add_edge(persist, out_objects)

    return graph


# 24
def solve_178fcbfb():
    arc_environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(arc_environment)

    # Constants
    red_1 = Constant(ColorValue, ColorValue(3))
    red_2 = Constant(ColorValue, ColorValue(3))
    center_1 = Constant(LocationValue, CENTERED)
    center_2 = Constant(LocationValue, CENTERED)
    z_index_non_red = Constant(int, 1)

    # Input nodes
    red_object = ArcInputNode(adjacency)
    non_red_object = ArcInputNode(adjacency)

    # Output nodes
    output_red = ArcOutputNode()
    output_non_red = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(red_object, equals, from_port="identity.color", to_port=0)
    graph.add_edge(red_1, equals, to_port=1)

    not_equals = NotEquals()
    graph.add_edge(non_red_object, not_equals, from_port="identity.color", to_port=0)
    graph.add_edge(red_2, not_equals, to_port=1)

    # Operators
    create_rectangle_1 = CreateRectangleOperator()
    graph.add_edge(red_object, create_rectangle_1, from_port="identity.color", to_port=0)
    graph.add_edge(red_object, create_rectangle_1, from_port="identity.shape.bounding_width", to_port=1)
    graph.add_edge(arc_environment.input_matrix_node(), create_rectangle_1, from_port="y_size", to_port=2)

    create_rectangle_2 = CreateRectangleOperator()
    graph.add_edge(non_red_object, create_rectangle_2, from_port="identity.color", to_port=0)
    graph.add_edge(arc_environment.input_matrix_node(), create_rectangle_2, from_port="x_size", to_port=1)
    graph.add_edge(non_red_object, create_rectangle_2, from_port="identity.shape.bounding_height", to_port=2)

    relocate_cardinal_1 = RelocateCardinalOperator()
    graph.add_edge(create_rectangle_1, relocate_cardinal_1, to_port=0)
    graph.add_edge(red_object, relocate_cardinal_1, from_port="location.x_location", to_port=1)
    graph.add_edge(center_1, relocate_cardinal_1, from_port="y_location", to_port=2)
    graph.add_edge(relocate_cardinal_1, output_red)

    relocate_cardinal_2 = RelocateCardinalOperator()
    graph.add_edge(create_rectangle_2, relocate_cardinal_2, to_port=0)
    graph.add_edge(center_2, relocate_cardinal_2, from_port="x_location", to_port=1)
    graph.add_edge(non_red_object, relocate_cardinal_2, from_port="location.y_location", to_port=2)

    set_z_index_non_red = SetZIndexOperator()
    graph.add_edge(relocate_cardinal_2, set_z_index_non_red, to_port=0)
    graph.add_edge(z_index_non_red, set_z_index_non_red, to_port=1)
    graph.add_edge(set_z_index_non_red, output_non_red)

    return graph


# 25
def solve_1a07d186():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)

    # Input nodes
    
    dot = ArcInputNode(adjacency)
    line = ArcInputNode(adjacency)

    # Output nodes
    output_dot = ArcOutputNode()
    output_line = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_1 = Equals()
    graph.add_edge(dot, equals_1, from_port="identity.shape.count", to_port=0)
    graph.add_edge(one, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(dot, equals_2, from_port="identity.color", to_port=0)
    graph.add_edge(line, equals_2, from_port="identity.color", to_port=1)

    not_equals = NotEquals()
    graph.add_edge(line, not_equals, from_port="identity.shape.count", to_port=0)
    graph.add_edge(one, not_equals, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(line, persist)
    graph.add_edge(persist, output_line)

    approach = ApproachOperator()
    graph.add_edge(dot, approach, to_port=0)
    graph.add_edge(line, approach, to_port=1)
    graph.add_edge(approach, output_dot)

    return graph


# 26
def solve_1b2d62fb():
    environment = ArcEnvironment([color_based_dividers_factory(2)], 1)
    graph = ArcGraph(environment)

    # Constants
    blue = Constant(ColorValue, ColorValue(2))
    light_blue = Constant(ColorValue, ColorValue(9))
    center = Constant(LocationValue, CENTERED)
    identity_type = Constant(Type, ObjectIdentity)

    # Input nodes
    rhs = ArcInputNode(color_based_dividers)
    lhs = ArcInputNode(color_based_dividers)

    # Output nodes
    output = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(rhs, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    not_eq_1 = NotEquals()
    graph.add_edge(rhs, not_eq_1, from_port="identity.color", to_port=0)
    graph.add_edge(blue, not_eq_1, to_port=1)

    not_eq_2 = NotEquals()
    graph.add_edge(lhs, not_eq_2, from_port="identity.color", to_port=0)
    graph.add_edge(blue, not_eq_2, to_port=1)

    # Operators
    and_op = BinaryNorOperator()
    graph.add_edge(rhs, and_op, from_port="identity", to_port=0)
    graph.add_edge(lhs, and_op, from_port="identity", to_port=1)

    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(light_blue, identity_constructor, to_port=1)
    graph.add_edge(and_op, identity_constructor, from_port="shape", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(identity_constructor, relocate, to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, output)

    return graph


# 27
def solve_1b60fb0c():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    identity_type = Constant(Type, ObjectIdentity)
    one = Constant(int, 1)
    red = Constant(ColorValue, ColorValue(3))
    actual_max = Constant(LocationSpecificAttribute, LocationSpecificAttribute.ACTUAL_MAX)
    z_index_blue = Constant(int, 1)

    # Inputs
    
    blue_object = ArcInputNode(adjacency)

    # Outputs
    blue_out = ArcOutputNode()
    red_out = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    pass

    # Operators
    persist = PersistOperator()
    graph.add_edge(blue_object, persist)

    set_z_index_blue = SetZIndexOperator()
    graph.add_edge(persist, set_z_index_blue, to_port=0)
    graph.add_edge(z_index_blue, set_z_index_blue, to_port=1)
    graph.add_edge(set_z_index_blue, blue_out)

    rotate = Rotate90Operator()
    graph.add_edge(blue_object, rotate, from_port="identity", to_port=0)
    graph.add_edge(one, rotate, to_port=1)

    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(red, identity_constructor, to_port=1)
    graph.add_edge(rotate, identity_constructor, from_port="shape", to_port=2)

    specific_x = SpecificLocationValueOperator()
    graph.add_edge(blue_object, specific_x, from_port="location.x_location", to_port=0)
    graph.add_edge(actual_max, specific_x, to_port=1)

    specific_y = SpecificLocationValueOperator()
    graph.add_edge(blue_object, specific_y, from_port="location.y_location", to_port=0)
    graph.add_edge(actual_max, specific_y, to_port=1)

    relocate = RelocateCardinalOperator()
    graph.add_edge(identity_constructor, relocate, to_port=0)
    graph.add_edge(specific_x, relocate, to_port=1)
    graph.add_edge(specific_y, relocate, to_port=2)
    graph.add_edge(relocate, red_out)

    return graph


# 28
def solve_1bfc4729():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    two = Constant(int, 2)
    seven = Constant(int, 7)
    top_shape = Constant(ArcObject, ArcObject([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    bot_shape = Constant(ArcObject, ArcObject([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))

    # Inputs
    
    dot = ArcInputNode(adjacency)

    # Outputs
    dot_out = ArcOutputNode()

    # Env shape
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(dot, equals, from_port="location.y_location.actual_min", to_port=0)
    graph.add_generic_edge([two, seven], equals, to_port=1)

    # Operators
    recolor = RecolorOperator()
    graph.add_generic_edge([top_shape, bot_shape], recolor, to_port=0)
    graph.add_edge(dot, recolor, from_port="identity.color", to_port=1)
    graph.add_edge(recolor, dot_out)

    return graph


# 29
def solve_1c786137():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    center = Constant(LocationValue, CENTERED)

    # Inputs
    selector_obj = ArcInputNode(adjacency)
    individual_obj = ArcInputNode(adjacency)
    objects_by_color = InputSetNode(ArcObject, perception_qualifiers={adjacency}, group_by="identity.color")

    # Outputs
    cluster_output = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(cluster_output, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    equals = Equals()
    graph.add_edge(objects_by_color, equals, from_port="size", to_port=0)
    graph.add_edge(one, equals, to_port=1)

    set_contains = SetContains()
    graph.add_edge(objects_by_color, set_contains, to_port=0)
    graph.add_edge(selector_obj, set_contains, to_port=1)

    obj_contains = ObjectContains()
    graph.add_edge(selector_obj, obj_contains, to_port=0)
    graph.add_edge(individual_obj, obj_contains, to_port=1)

    inner_object_set = SetJoin()
    graph.add_edge(individual_obj, inner_object_set)

    set_to_cluster = CreateCompositeObjectOperator()
    graph.add_edge(inner_object_set, set_to_cluster)

    relocate = RelocateOperator()
    graph.add_edge(set_to_cluster, relocate, from_port="identity", to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, cluster_output)

    return graph


# 30
def solve_1caeab9d():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    blue = Constant(ColorValue, ColorValue(2))

    # Input nodes
    
    blue_object = ArcInputNode(adjacency)
    non_blue_object = ArcInputNode(adjacency)

    # Output nodes
    output_blue = ArcOutputNode()
    output_non_blue = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(blue_object, equals, from_port="identity.color", to_port=0)
    graph.add_edge(blue, equals, to_port=1)

    not_equals = NotEquals()
    graph.add_edge(non_blue_object, not_equals, from_port="identity.color", to_port=0)
    graph.add_edge(blue, not_equals, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(blue_object, persist)
    graph.add_edge(persist, output_blue)

    relocate_cardinal = RelocateCardinalOperator()
    graph.add_edge(non_blue_object, relocate_cardinal, from_port="identity", to_port=0)
    graph.add_edge(non_blue_object, relocate_cardinal, from_port="location.x_location", to_port=1)
    graph.add_edge(blue_object, relocate_cardinal, from_port="location.y_location", to_port=2)
    graph.add_edge(relocate_cardinal, output_non_blue)

    return graph


# 31
def solve_1cf80156():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    object_node = ArcInputNode(adjacency)

    # Output nodes
    output_object = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(object_node, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Operators
    relocate = RelocateOperator()
    graph.add_edge(object_node, relocate, from_port="identity", to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, output_object)

    return graph


# 32
def solve_1e0a9b12():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    y = Constant(Direction, Direction.DIRECTION_Y)
    positive = Constant(int, 1)

    # Inputs
    
    all_objects = InputSetNode(ArcObject, perception_qualifiers={adjacency})

    # Outputs
    output_objects = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    pass

    # Operators
    collapse = CollapseDirectionOperator()
    graph.add_edge(all_objects, collapse, to_port=0)
    graph.add_edge(y, collapse, to_port=1)
    graph.add_edge(positive, collapse, to_port=2)

    split_output_set = SetSplit(ArcObject)
    graph.add_edge(collapse, split_output_set)
    graph.add_edge(split_output_set, output_objects)

    return graph


# 33
def solve_1e32b0e9():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1) # constant shape would be more acurate here.
    identity_type = Constant(Type, ObjectIdentity)
    loc_1_filter = Constant(LocationValue,
                            LocationValue(LinearLocation(center=0.14705882352941177),
                                          LinearLocation(center=0.14705882352941177)))
    loc_1 = Constant(LocationValue, LocationValue(LinearLocation(center=0.14705882352941177),
                                                  LinearLocation(center=0.14705882352941177)))
    loc_2 = Constant(LocationValue, LocationValue(LinearLocation(center=0.5),
                                                  LinearLocation(center=0.14705882352941177)))
    loc_3 = Constant(LocationValue, LocationValue(LinearLocation(center=0.8529411764705882),
                                                  LinearLocation(center=0.14705882352941177)))
    loc_4 = Constant(LocationValue, LocationValue(LinearLocation(center=0.14705882352941177),
                                                  LinearLocation(center=0.5)))
    loc_5 = Constant(LocationValue, LocationValue(LinearLocation(center=0.5),
                                                  LinearLocation(center=0.5)))
    loc_6 = Constant(LocationValue, LocationValue(LinearLocation(center=0.8529411764705882),
                                                  LinearLocation(center=0.5)))
    loc_7 = Constant(LocationValue, LocationValue(LinearLocation(center=0.14705882352941177),
                                                  LinearLocation(center=0.8529411764705882)))
    loc_8 = Constant(LocationValue, LocationValue(LinearLocation(center=0.5),
                                                  LinearLocation(center=0.8529411764705882)))
    loc_9 = Constant(LocationValue, LocationValue(LinearLocation(center=0.8529411764705882),
                                                  LinearLocation(center=0.8529411764705882)))
    z_index_persisted = Constant(int, 1)

    # Inputs
    
    persisted_input = ArcInputNode(adjacency)
    fence = ArcInputNode(adjacency)
    key_obj = ArcInputNode(adjacency)

    # Outputs
    persisted_out = ArcOutputNode()
    copied_out = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_1 = Equals()
    graph.add_edge(fence, equals_1, from_port="location.x_location.boundary", to_port=0)
    graph.add_edge(one, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(fence, equals_2, from_port="location.y_location.boundary", to_port=0)
    graph.add_edge(one, equals_2, to_port=1)

    equals_3 = Equals()
    graph.add_edge(key_obj, equals_3, from_port="location", to_port=0)
    graph.add_edge(loc_1_filter, equals_3, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(persisted_input, persist)

    set_z_index_persisted = SetZIndexOperator()
    graph.add_edge(persist, set_z_index_persisted, to_port=0)
    graph.add_edge(z_index_persisted, set_z_index_persisted, to_port=1)
    graph.add_edge(set_z_index_persisted, persisted_out)

    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(fence, identity_constructor, from_port="identity.color", to_port=1)
    graph.add_edge(key_obj, identity_constructor, from_port="identity.shape", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(identity_constructor, relocate, to_port=0)
    graph.add_generic_edge([loc_1, loc_2, loc_3, loc_4, loc_5, loc_6, loc_7, loc_8, loc_9], relocate, to_port=1)
    graph.add_edge(relocate, copied_out)

    return graph


# 34
def solve_1f0c79e5():
    environment = ArcEnvironment([adjacency_any_color, individual_cells], 1)
    graph = ArcGraph(environment)

    # Constants
    red = Constant(ColorValue, ColorValue(3))

    tri_1 = Constant(ArcObject, ArcObject([[1, 1],
                                           [1, 1]]))
    tri_2 = Constant(ArcObject, ArcObject([[1, 0],
                                           [1, 1]]))
    tri_3 = Constant(ArcObject, ArcObject([[1, 1],
                                           [0, 1]]))
    tri_4 = Constant(ArcObject, ArcObject([[1, 1],
                                           [1, 0]]))
    first_rel = Constant(TwoDimensionalLocationRelationship, TwoDimensionalLocationRelationship(OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_X, None, -1),
                        OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_Y, None, -1)))
    second_rel = Constant(TwoDimensionalLocationRelationship, TwoDimensionalLocationRelationship(OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_X, None, -1),
                        OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_Y, None, 1)))
    third_rel =Constant(TwoDimensionalLocationRelationship, TwoDimensionalLocationRelationship(OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_X, None, 1),
                        OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_Y, None, -1)))
    fourth_rel = Constant(TwoDimensionalLocationRelationship, TwoDimensionalLocationRelationship(OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_X, None, 1),
                        OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_Y, None, 1)))
    z_index_color_dot = Constant(int, 2)
    z_index_placed_tri = Constant(int, 1)

    # Input nodes
    square_obj = ArcInputNode(adjacency_any_color)
    red_dot = ArcInputNode(individual_cells)
    non_red_dot = ArcInputNode(individual_cells)

    # Output node
    color_box = ArcOutputNode()
    placed_tri = ArcOutputNode()
    color_dot = ArcOutputNode()

    # Matrix size
    pass

    # relationships
    equals_color = Equals()
    graph.add_edge(red_dot, equals_color, from_port="identity.color", to_port=0)
    graph.add_edge(red, equals_color, to_port=1)

    not_equals_color_1 = NotEquals()
    graph.add_edge(non_red_dot, not_equals_color_1, from_port="identity.color", to_port=0)
    graph.add_edge(red, not_equals_color_1, to_port=1)

    equals_x = Equals()
    graph.add_generic_edge(red_dot, equals_x, from_port=["location.x_location.actual_min", "location.x_location.actual_min",
                                                         "location.x_location.actual_max", "location.x_location.actual_max"], to_port=0)
    graph.add_generic_edge(square_obj, equals_x, from_port=["location.x_location.actual_min", "location.x_location.actual_min",
                                                            "location.x_location.actual_max", "location.x_location.actual_max"], to_port=1)

    equals_y = Equals()
    graph.add_generic_edge(red_dot, equals_y, from_port=["location.y_location.actual_min", "location.y_location.actual_max",
                                                         "location.y_location.actual_min", "location.y_location.actual_max"], to_port=0)
    graph.add_generic_edge(square_obj, equals_y, from_port=["location.y_location.actual_min", "location.y_location.actual_max",
                                                            "location.y_location.actual_min", "location.y_location.actual_max"], to_port=1)


    # Operators
    persist = PersistOperator()
    graph.add_edge(square_obj, persist)
    graph.add_edge(persist, color_box)

    recursive_proxy = RecursiveProxyNode()
    graph.add_edge(square_obj, recursive_proxy, to_port=0)
    graph.add_edge(placed_tri, recursive_proxy, to_port=1)

    recolor = RecolorOperator()
    graph.add_generic_edge([tri_1, tri_3, tri_2, tri_4], recolor, to_port=0)
    graph.add_edge(non_red_dot, recolor, from_port="identity.color", to_port=1)

    recolor_dot = RecolorOperator()
    graph.add_generic_edge(red_dot, recolor_dot, to_port=0)
    graph.add_edge(non_red_dot, recolor_dot, from_port="identity.color", to_port=1)

    set_z_index_color_dot = SetZIndexOperator()
    graph.add_edge(recolor_dot, set_z_index_color_dot, to_port=0)
    graph.add_edge(z_index_color_dot, set_z_index_color_dot, to_port=1)
    graph.add_edge(set_z_index_color_dot, color_dot)

    apply_location = ApplyTwoDimensionalRelationshipOperator()
    graph.add_generic_edge([first_rel, second_rel, third_rel, fourth_rel], apply_location, to_port=0)
    graph.add_edge(recursive_proxy, apply_location, to_port=1)

    relocate = RelocateOperator()
    graph.add_edge(recolor, relocate, from_port="identity", to_port=0)
    graph.add_edge(apply_location, relocate, to_port=1)

    set_z_index_placed_tri = SetZIndexOperator()
    graph.add_edge(relocate, set_z_index_placed_tri, to_port=0)
    graph.add_edge(z_index_placed_tri, set_z_index_placed_tri, to_port=1)
    graph.add_edge(set_z_index_placed_tri, placed_tri)

    return graph


# 35
def solve_1f642eb9():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    light_blue = Constant(ColorValue, ColorValue(9))
    dir_x = Constant(Direction, Direction.DIRECTION_X)
    dir_y = Constant(Direction, Direction.DIRECTION_Y)
    z_index_dot_moved = Constant(int, 1)

    # Inputs
    
    blue_object = ArcInputNode(adjacency)
    dot = ArcInputNode(adjacency)

    # Outputs
    blue_object_out = ArcOutputNode()
    dot_moved = ArcOutputNode()
    dot_out = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(blue_object, equals, from_port="identity.color", to_port=0)
    graph.add_edge(light_blue, equals, to_port=1)

    not_equals = NotEquals()
    graph.add_edge(dot, not_equals, from_port="identity.color", to_port=0)
    graph.add_edge(light_blue, not_equals, to_port=1)

    equals_2 = Equals()
    graph.add_generic_edge(dot, equals_2, from_port=["location.x_location.min_boundary", "location.x_location.max_boundary",
                                                     "location.y_location.min_boundary", "location.y_location.max_boundary"], to_port=0)
    graph.add_edge(one, equals_2, to_port=1)

    # Operators
    # perist_1 = PersistOperator()
    graph.add_edge(blue_object, blue_object_out)

    # perist_2 = PersistOperator()
    graph.add_edge(dot, dot_out)

    minus = SubtractionOperator()
    graph.add_generic_edge(blue_object, minus, from_port=["location.x_location.actual_min", "location.x_location.actual_max",
                                                         "location.y_location.actual_min", "location.y_location.actual_max"], to_port=0)
    graph.add_generic_edge(dot, minus, from_port=["location.x_location.actual_min", "location.x_location.actual_max",
                                                  "location.y_location.actual_min", "location.y_location.actual_max"], to_port=1)

    move = MoveOperator()
    graph.add_edge(dot, move, to_port=0)
    graph.add_generic_edge([dir_x, dir_x, dir_y, dir_y], move, to_port=1)
    graph.add_edge(minus, move, to_port=2)

    set_z_index_dot_moved = SetZIndexOperator()
    graph.add_edge(move, set_z_index_dot_moved, to_port=0)
    graph.add_edge(z_index_dot_moved, set_z_index_dot_moved, to_port=1)
    graph.add_edge(set_z_index_dot_moved, dot_moved)

    return graph


# 36
def solve_1f85a75f():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    color_set = InputSetNode(ArcObject, perception_qualifiers={adjacency}, group_by="identity.color")

    # Output nodes
    output_object = ArcOutputNode()

    main_object = SetSplit(ArcObject)
    graph.add_edge(color_set, main_object)

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(main_object, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    equals = Equals()
    graph.add_edge(color_set, equals, from_port="size")
    graph.add_edge(one, equals, to_port=1)

    # Operators
    relocate = RelocateOperator()
    graph.add_edge(main_object, relocate, from_port="identity", to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, output_object)

    return graph


# 37
def solve_1f876c06():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    pass

    # Inputs
    
    first_object = ArcInputNode(adjacency)
    second_object = ArcInputNode(adjacency)

    # Output nodes
    line_object = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(first_object, equals, from_port="identity.color", to_port=0)
    graph.add_edge(second_object, equals, from_port="identity.color", to_port=1)

    # Operators
    draw_line = DrawLineOperator()
    graph.add_edge(first_object, draw_line, from_port="identity.color", to_port=0)
    graph.add_edge(first_object, draw_line, from_port="location", to_port=1)
    graph.add_edge(second_object, draw_line, from_port="location", to_port=2)
    graph.add_edge(draw_line, line_object)

    return graph


# 38
def solve_1fad071e():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    x_loc = Constant(LinearLocation, LinearLocation(actual_min=0))
    center = Constant(LocationValue, CENTERED)
    blue_square_const = Constant(ArcObject, ArcObject([[2, 2],
                                                       [2, 2]]))
    output_shape = Constant(ArcObject, ArcObject([[1, 1, 1, 1, 1]]))

    # Input nodes
    blue_square = ArcInputNode(adjacency)

    # Output nodes
    output_rectangle = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(output_shape, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    equals_1 = Equals()
    graph.add_edge(blue_square, equals_1, from_port="identity", to_port=0)
    graph.add_edge(blue_square_const, equals_1, from_port="identity", to_port=1)

    set_member = SetJoin()
    graph.add_edge(blue_square, set_member)

    # Operators
    create_rectangle = CreateRectangleOperator()
    graph.add_edge(blue_square, create_rectangle, from_port="identity.color", to_port=0)
    graph.add_edge(set_member, create_rectangle, from_port="size", to_port=1)
    graph.add_edge(output_shape, create_rectangle, from_port="identity.shape.bounding_height", to_port=2)

    relocate = RelocateCardinalOperator()
    graph.add_edge(create_rectangle, relocate, to_port=0)
    graph.add_edge(x_loc, relocate, to_port=1)
    graph.add_edge(center, relocate, from_port="y_location", to_port=2)
    graph.add_edge(relocate, output_rectangle)

    return graph


# 39
# TODO re-enable when [] is removed
def solve_x_2013d3e2():

    environment = ArcEnvironment([adjacency_any_color], 1)
    graph = ArcGraph(environment)

    # Constants
    two = Constant(int, 2)
    center = Constant(LocationValue, CENTERED)

    # Inputs
    thing = ArcInputNode(adjacency_any_color)

    # Outputs
    out_section = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(out_section, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    pass

    # Operators
    divide_identity = DivideIdentityOperator()
    graph.add_edge(thing, divide_identity, from_port="identity", to_port=0)
    graph.add_edge(two, divide_identity, to_port=1)
    graph.add_edge(two, divide_identity, to_port=2)

    relocate = RelocateOperator()
    # TODO USECASE FOR RUNTIME RELATIONSHIP EVAL - USE ZONE INFO TO TELL WHICH OBJECT NONE OF THIS SET INDEX STUFF
    graph.add_edge(divide_identity, relocate, from_port="[0]", to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, out_section)

    return graph


# 40
def solve_2204b7a8():
    environment = ArcEnvironment([adjacency_cardinal], 1)
    graph = ArcGraph(environment)

    # Constants
    one_1 = Constant(int, 1)
    one_2 = Constant(int, 1)
    two = Constant(int, 2)

    # Input objects
    
    line = ArcInputNode(adjacency_cardinal)
    line_set = InputSetNode(ArcObject, perception_qualifiers={adjacency_cardinal}, group_by="identity.shape")
    dot = ArcInputNode(adjacency_cardinal)

    # Output objects
    out_line = ArcOutputNode()
    out_dot = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    not_equals = NotEquals()
    graph.add_edge(line, not_equals, from_port="identity.shape.count", to_port=0)
    graph.add_edge(one_1, not_equals, to_port=1)

    equals_1 = Equals()
    graph.add_edge(dot, equals_1, from_port="identity.shape.count", to_port=0)
    graph.add_edge(one_2, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(line_set, equals_2, from_port="size", to_port=0)
    graph.add_edge(two, equals_2, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(line, persist)
    graph.add_edge(persist, out_line)

    closest_line = FindClosestOperator()
    graph.add_edge(line_set, closest_line, to_port=0)
    graph.add_edge(dot, closest_line, to_port=1)

    recolor = RecolorOperator()
    graph.add_edge(dot, recolor, to_port=0)
    graph.add_edge(closest_line, recolor, from_port="identity.color", to_port=1)
    graph.add_edge(recolor, out_dot)

    return graph


# 41
def solve_22168020():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    pass

    # Inputs
    thing = ArcInputNode(adjacency)

    # Outputs
    out_thing = ArcOutputNode()
    out_cavity = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    get_cavities = GetOuterCavitiesOperator()
    graph.add_edge(thing, get_cavities)

    top_cavity = SetSplit(ArcObject)
    graph.add_edge(get_cavities, top_cavity)

    equals_2 = Equals()
    graph.add_edge(thing, equals_2, from_port="location.x_location", to_port=0)
    graph.add_edge(top_cavity, equals_2, from_port="location.x_location", to_port=1)

    equals_3 = Equals()
    graph.add_edge(thing, equals_3, from_port="location.y_location.actual_min", to_port=0)
    graph.add_edge(top_cavity, equals_3, from_port="location.y_location.actual_min", to_port=1)

    # Operators
    graph.add_edge(thing, out_thing)

    recolor = RecolorOperator()
    graph.add_edge(top_cavity, recolor, to_port=0)
    graph.add_edge(thing, recolor, from_port="identity.color", to_port=1)
    graph.add_edge(recolor, out_cavity)

    return graph


# 42
def solve_22233c11():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    # NOTE these objects are store with extra space so I don't have to write the relationship vector constant verbatim...
    norm_object = Constant(ArcObject, ArcObject([[0, 0, 0, 0],
                                                 [0, 4, 0, 0],
                                                 [0, 0, 4, 0],
                                                 [0, 0, 0, 0]]))
    other_object_1 = Constant(ArcObject, ArcObject([[0, 0, 0, 9],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0]]))
    other_object_2 = Constant(ArcObject, ArcObject([[0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [9, 0, 0, 0]]))
    const_loc_rel_1 = Constant(TwoDimensionalLocationRelationship,
                               MeasureCentersOperator().apply(norm_object.value, other_object_1.value))
    const_loc_rel_2 = Constant(TwoDimensionalLocationRelationship,
                               MeasureCentersOperator().apply(norm_object.value, other_object_2.value))

    # Input objects
    
    main_node = ArcInputNode(adjacency)

    # Output objects
    out_main_node = ArcOutputNode()
    out_new_node = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    # equals = Equals()
    # graph.add_edge(main_node, equals, from_port="identity.shape.normalized_shape", to_port=0)
    # graph.add_edge(norm_object, equals, from_port="identity.shape.normalized_shape", to_port=1)

    abstract_identity_relationship = MeasureIdentityDeltaOperator()
    graph.add_edge(norm_object, abstract_identity_relationship, from_port="identity", to_port=0)
    graph.add_edge(main_node, abstract_identity_relationship, from_port="identity", to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(main_node, persist)
    graph.add_edge(persist, out_main_node)

    apply_id_rel = ApplyIdentityRelationshipOperator()
    graph.add_edge(abstract_identity_relationship, apply_id_rel, to_port=0)
    graph.add_generic_edge([other_object_1, other_object_2], apply_id_rel, from_port="identity", to_port=1)

    transform_loc = TransformLocationRelationshipOperator()
    graph.add_edge(abstract_identity_relationship, transform_loc, to_port=0)
    graph.add_generic_edge([const_loc_rel_1, const_loc_rel_2], transform_loc, to_port=1)

    apply_loc = ApplyTwoDimensionalRelationshipOperator()
    graph.add_edge(transform_loc, apply_loc, to_port=0)
    graph.add_edge(main_node, apply_loc, to_port=1)

    relocate = RelocateOperator()
    graph.add_edge(apply_id_rel, relocate, to_port=0)
    graph.add_edge(apply_loc, relocate, to_port=1)
    graph.add_edge(relocate, out_new_node)

    return graph


# 43
def solve_2281f1f4():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    red = Constant(ColorValue, ColorValue(3))

    # Input nodes
    
    top_node = ArcInputNode(adjacency)
    side_node = ArcInputNode(adjacency)

    # Output nodes
    output_top = ArcOutputNode()
    output_side = ArcOutputNode()
    output_red = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(top_node, equals, from_port="location.y_location.min_boundary", to_port=0)
    graph.add_edge(one, equals, to_port=1)

    equals_2 = Equals()
    graph.add_edge(side_node, equals_2, from_port="location.x_location.max_boundary", to_port=0)
    graph.add_edge(one, equals_2, to_port=1)

    # Operators
    persist_1 = PersistOperator()
    graph.add_edge(top_node, persist_1)
    graph.add_edge(persist_1, output_top)

    persist_2 = PersistOperator()
    graph.add_edge(side_node, persist_2)
    graph.add_edge(persist_2, output_side)

    rectangle_identity = CreateRectangleOperator()
    graph.add_edge(red, rectangle_identity, to_port=0)
    graph.add_edge(top_node, rectangle_identity, from_port="identity.shape.bounding_width", to_port=1)
    graph.add_edge(side_node, rectangle_identity, from_port="identity.shape.bounding_height", to_port=2)

    relocate = RelocateCardinalOperator()
    graph.add_edge(rectangle_identity, relocate, to_port=0)
    graph.add_edge(top_node, relocate, from_port="location.x_location", to_port=1)
    graph.add_edge(side_node, relocate, from_port="location.y_location", to_port=2)
    graph.add_edge(relocate, output_red)

    return graph


# 44
def solve_228f6490():
    environment = ArcEnvironment([adjacency, inner_cavities], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    grey = Constant(ColorValue, ColorValue(6))

    # Input nodes
    
    grey_object = ArcInputNode(adjacency)
    cavity_object = ArcInputNode(inner_cavities)
    moving_object = ArcInputNode(adjacency)
    non_moving_object = ArcInputNode(adjacency)
    color_set = InputSetNode(ArcObject, perception_qualifiers={adjacency}, group_by="identity.color")
    color_set_2 = InputSetNode(ArcObject, perception_qualifiers={adjacency}, group_by="identity.color")

    # Output nodes
    grey_object_out = ArcOutputNode()
    moving_object_out = ArcOutputNode()
    non_moving_object_out = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_1 = Equals()
    graph.add_edge(grey_object, equals_1, from_port="identity.color", to_port=0)
    graph.add_edge(grey, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(moving_object, equals_2, from_port="identity.shape", to_port=0)
    graph.add_edge(cavity_object, equals_2, from_port="identity.shape", to_port=1)

    equals_3 = Equals()
    graph.add_edge(color_set, equals_3, from_port="size", to_port=0)
    graph.add_edge(one, equals_3, to_port=1)

    set_contains_1 = SetContains()
    graph.add_edge(color_set, set_contains_1, to_port=0)
    graph.add_edge(moving_object, set_contains_1, to_port=1)

    not_equals = NotEquals()
    graph.add_edge(color_set_2, not_equals, from_port="size", to_port=0)
    graph.add_edge(one, not_equals, to_port=1)

    set_contains_2 = SetContains()
    graph.add_edge(color_set_2, set_contains_2, to_port=0)
    graph.add_edge(non_moving_object, set_contains_2, to_port=1)

    # Operators
    persist_1 = PersistOperator()
    graph.add_edge(grey_object, persist_1)
    graph.add_edge(persist_1, grey_object_out)

    persist_2 = PersistOperator()
    graph.add_edge(non_moving_object, persist_2)
    graph.add_edge(persist_2, non_moving_object_out)

    relocate = RelocateOperator()
    graph.add_edge(moving_object, relocate, from_port="identity", to_port=0)
    graph.add_edge(cavity_object, relocate, from_port="location", to_port=1)
    graph.add_edge(relocate, moving_object_out)

    return graph


# 45
def solve_22eb0ac0():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    pass

    # Input nodes
    
    right_node = ArcInputNode(adjacency)
    left_node = ArcInputNode(adjacency)
    other_node = ArcInputNode(adjacency)

    # Output nodes
    rectangle_output = ArcOutputNode()
    other_node_output = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_1 = Equals()
    graph.add_edge(right_node, equals_1, from_port="location.y_location.center", to_port=0)
    graph.add_edge(left_node, equals_1, from_port="location.y_location.center", to_port=1)

    equals_2 = Equals()
    graph.add_edge(right_node, equals_2, from_port="identity.color", to_port=0)
    graph.add_edge(left_node, equals_2, from_port="identity.color", to_port=1)

    # Operators
    draw_line = DrawLineOperator()
    graph.add_edge(left_node, draw_line, from_port="identity.color", to_port=0)
    graph.add_edge(left_node, draw_line, from_port="location", to_port=1)
    graph.add_edge(right_node, draw_line, from_port="location", to_port=2)
    graph.add_edge(draw_line, rectangle_output)

    persist = PersistOperator()
    graph.add_edge(other_node, persist)
    graph.add_edge(persist, other_node_output)

    return graph

# 46
# TODO need a new way to do this without parent id.
def solve_x_234bbc79():
    environment = ArcEnvironment([adjacency_any_color, adjacency_cardinal], 1)
    graph = ArcGraph(environment)

    # Constants
    identity = Constant(Type, ObjectIdentity)
    env_shape_type = Constant(Type, EnvironmentShape)
    one_1 = Constant(int, 1)
    one_2 = Constant(int, 1)
    grey_1 = Constant(ColorValue, ColorValue(6))
    grey_2 = Constant(ColorValue, ColorValue(6))
    constant_delta_rel = Constant(TwoDimensionalLocationRelationship, TwoDimensionalLocationRelationship(
        OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_X, None, 1, False),
        OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_Y, None, 0, True),
    ))

    # Inputs
    object_set = InputSetNode(ArcObject, perception_qualifiers={adjacency_any_color})
    object_set_size = InputSetNode(ArcObject, perception_qualifiers={adjacency_any_color})
    rightmost = ArcInputNode(adjacency_any_color)
    rightmost_color_child = ArcInputNode(adjacency_cardinal)
    starter = ArcInputNode(adjacency_any_color)
    starter_right_child = ArcInputNode(adjacency_cardinal)
    target = ArcInputNode(adjacency_any_color)
    target_color_child = ArcInputNode(adjacency_cardinal)
    target_left_child = ArcInputNode(adjacency_cardinal)

    # Outputs
    out_rightmost = ArcOutputNode()
    out_target = ArcOutputNode()

    # Matrix size
    sum_set = SumSetOperator()
    graph.add_edge(object_set_size, sum_set, from_port="*.identity.shape.bounding_width")

    env_shape_constructor = ConstructorOperator()
    graph.add_edge(env_shape_type, env_shape_constructor, to_port=0)
    graph.add_edge(environment.input_matrix_node(), env_shape_constructor, from_port="y_size", to_port=1)
    graph.add_edge(sum_set, env_shape_constructor, to_port=2)
    graph.add_edge(env_shape_constructor, environment.output_matrix_node)

    # Relationships
    equals_1 = Equals()
    graph.add_edge(rightmost, equals_1, from_port="location.x_location.min_boundary", to_port=0)
    graph.add_edge(one_1, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(rightmost, equals_2, from_port="id", to_port=0)
    graph.add_edge(rightmost_color_child, equals_2, from_port="parent_id", to_port=1)

    not_equals_1 = NotEquals()
    graph.add_edge(rightmost_color_child, not_equals_1, from_port="identity.color", to_port=0)
    graph.add_edge(grey_1, not_equals_1, to_port=1)

    set_rank_1 = SetRankOperator()
    graph.add_edge(object_set, set_rank_1, from_port="*.location.x_location.center", to_port=0)
    graph.add_edge(starter, set_rank_1, from_port="location.x_location.center", to_port=1)

    set_rank_2 = SetRankOperator()
    graph.add_edge(object_set, set_rank_2, from_port="*.location.x_location.center", to_port=0)
    graph.add_edge(target, set_rank_2, from_port="location.x_location.center", to_port=1)

    runtime_order = RuntimeOrderNode()
    graph.add_edge(set_rank_1, runtime_order, to_port=0)

    addition = AdditionOperator()
    graph.add_edge(set_rank_1, addition, to_port=0)
    graph.add_edge(one_2, addition, to_port=1)

    equals_3 = Equals()
    graph.add_edge(addition, equals_3, to_port=0)
    graph.add_edge(set_rank_2, equals_3, to_port=1)

    equals_4 = Equals()
    graph.add_edge(starter, equals_4, from_port="id", to_port=0)
    graph.add_edge(starter_right_child, equals_4, from_port="parent_id", to_port=1)

    equals_5 = Equals()
    graph.add_edge(starter_right_child, equals_5, from_port="identity.color", to_port=0)
    graph.add_edge(grey_2, equals_5, to_port=1)

    equals_5_2 = Equals()
    graph.add_edge(starter, equals_5_2, from_port="location.x_location.actual_max", to_port=0)
    graph.add_edge(starter_right_child, equals_5_2, from_port="location.x_location.actual_max", to_port=1)

    equals_6 = Equals()
    graph.add_edge(target, equals_6, from_port="id", to_port=0)
    graph.add_edge(target_left_child, equals_6, from_port="parent_id", to_port=1)

    equals_7 = Equals()
    graph.add_edge(target_left_child, equals_7, from_port="identity.color", to_port=0)
    graph.add_edge(grey_2, equals_7, to_port=1)

    equals_8 = Equals()
    graph.add_edge(target, equals_8, from_port="location.x_location.actual_min", to_port=0)
    graph.add_edge(target_left_child, equals_8, from_port="location.x_location.actual_min", to_port=1)

    equals_9 = Equals()
    graph.add_edge(target, equals_9, from_port="id", to_port=0)
    graph.add_edge(target_color_child, equals_9, from_port="parent_id", to_port=1)

    not_equals_2 = NotEquals()
    graph.add_edge(target_color_child, not_equals_2, from_port="identity.color", to_port=0)
    graph.add_edge(grey_2, not_equals_2, to_port=1)

    # Recursive proxy
    iterative_proxy = IterativeProxyNode()
    graph.add_edge(starter, iterative_proxy, to_port=0)
    graph.add_edge(out_target, iterative_proxy, to_port=1)

    # Operators
    recolor = RecolorOperator()
    graph.add_edge(rightmost, recolor, to_port=0)
    graph.add_edge(rightmost_color_child, recolor, from_port="identity.color", to_port=1)
    graph.add_edge(recolor, out_rightmost)

    dist_1 = MeasureCentersOperator()
    graph.add_edge(target_left_child, dist_1, to_port=0)
    graph.add_edge(target, dist_1, to_port=1)

    dist_2 = MeasureCentersOperator()
    graph.add_edge(starter, dist_2, to_port=0)
    graph.add_edge(starter_right_child, dist_2, to_port=1)

    sum_dist_1 = SumLocationRelationshipOperator()
    graph.add_edge(dist_1, sum_dist_1, to_port=0)
    graph.add_edge(dist_2, sum_dist_1, to_port=1)

    sum_dist_2 = SumLocationRelationshipOperator()
    graph.add_edge(constant_delta_rel, sum_dist_2, to_port=0)
    graph.add_edge(sum_dist_1, sum_dist_2, to_port=1)

    apply_sum_relationship = ApplyTwoDimensionalRelationshipOperator()
    graph.add_edge(sum_dist_2, apply_sum_relationship, to_port=0)
    graph.add_edge(iterative_proxy, apply_sum_relationship, to_port=1)

    compose_identity = ConstructorOperator()
    graph.add_edge(identity, compose_identity, to_port=0)
    graph.add_edge(target_color_child, compose_identity, from_port="identity.color", to_port=1)
    graph.add_edge(target, compose_identity, from_port="identity.shape", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(compose_identity, relocate, to_port=0)
    graph.add_edge(apply_sum_relationship, relocate, to_port=1)
    graph.add_edge(relocate, out_target)

    return graph


# 47
def solve_23581191():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    one = Constant(int, 1)
    center = Constant(LocationValue, CENTERED)
    red_dot = Constant(ArcObject, ArcObject([[3]]))
    z_index_red_dot = Constant(int, 1)


    # Input objects
    dot = ArcInputNode(adjacency)
    dot_1 = ArcInputNode(adjacency)
    dot_2 = ArcInputNode(adjacency)

    # Output nodes
    vertical_line = ArcOutputNode()
    horizontal_line = ArcOutputNode()
    red_dot_out = ArcOutputNode()

    # Matrix shape
    pass

    # Relationships
    pass

    # Operators
    create_vert_line = CreateRectangleOperator()
    graph.add_edge(dot, create_vert_line, from_port="identity.color", to_port=0)
    graph.add_edge(one, create_vert_line, to_port=1)
    graph.add_edge(environment.input_matrix_node(), create_vert_line, from_port="y_size", to_port=2)

    create_horz_line = CreateRectangleOperator()
    graph.add_edge(dot, create_horz_line, from_port="identity.color", to_port=0)
    graph.add_edge(environment.input_matrix_node(), create_horz_line, from_port="x_size", to_port=1)
    graph.add_edge(one, create_horz_line, to_port=2)

    place_vert_line = RelocateCardinalOperator()
    graph.add_edge(create_vert_line, place_vert_line, to_port=0)
    graph.add_edge(dot, place_vert_line, from_port="location.x_location", to_port=1)
    graph.add_edge(center, place_vert_line, from_port="y_location", to_port=2)
    graph.add_edge(place_vert_line, vertical_line)

    place_horz_line = RelocateCardinalOperator()
    graph.add_edge(create_horz_line, place_horz_line, to_port=0)
    graph.add_edge(center, place_horz_line, from_port="x_location", to_port=1)
    graph.add_edge(dot, place_horz_line, from_port="location.y_location", to_port=2)
    graph.add_edge(place_horz_line, horizontal_line)

    place_red_dot = RelocateCardinalOperator()
    graph.add_edge(red_dot, place_red_dot, from_port="identity", to_port=0)
    graph.add_edge(dot_1, place_red_dot, from_port="location.x_location", to_port=1)
    graph.add_edge(dot_2, place_red_dot, from_port="location.y_location", to_port=2)

    set_z_index_red_dot = SetZIndexOperator()
    graph.add_edge(place_red_dot, set_z_index_red_dot, to_port=0)
    graph.add_edge(z_index_red_dot, set_z_index_red_dot, to_port=1)
    graph.add_edge(set_z_index_red_dot, red_dot_out)

    return graph


# 48
def solve_239be575():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    red = Constant(ColorValue, ColorValue(3))
    blue = Constant(ColorValue, ColorValue(9))
    dot = Constant(ArcObject, ArcObject([[1]]))
    dot_other = Constant(ArcObject, ArcObject([[1]]))
    center = Constant(LocationValue, CENTERED)
    identity_type = Constant(Type, ObjectIdentity)

    # Input nodes
    red_1 = ArcInputNode(adjacency)
    red_2 = ArcInputNode(adjacency)
    blue_middle = ArcInputNode(adjacency)

    # Output nodes
    blue_square = ArcOutputNode()

    # Matrix size
    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(dot_other, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    equals_1 = Equals()
    graph.add_edge(red_1, equals_1, from_port="identity.color", to_port=0)
    graph.add_edge(red, equals_1, to_port=1)

    equals_2 = Equals()
    graph.add_edge(red_2, equals_2, from_port="identity.color", to_port=0)
    graph.add_edge(red, equals_2, to_port=1)

    equals_3 = Equals()
    graph.add_edge(blue_middle, equals_3, from_port="identity.color", to_port=0)
    graph.add_edge(blue, equals_3, to_port=1)

    touches_1 = Touches()
    graph.add_edge(red_1, touches_1, to_port=0)
    graph.add_edge(blue_middle, touches_1, to_port=1)

    touches_2 = Touches()
    graph.add_edge(red_2, touches_2, to_port=0)
    graph.add_edge(blue_middle, touches_2, to_port=1)

    # Operators
    identity_constructor = ConstructorOperator()
    graph.add_edge(identity_type, identity_constructor, to_port=0)
    graph.add_edge(blue_middle, identity_constructor, from_port="identity.color", to_port=1)
    graph.add_edge(dot, identity_constructor, from_port="identity.shape", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(identity_constructor, relocate, to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, blue_square)

    return graph


# 49
def solve_23b5c85d():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    zero = Constant(int, 0)
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    object_node = ArcInputNode(adjacency)
    object_set = InputSetNode(ArcObject, perception_qualifiers={adjacency})

    # Output nodes
    output_object = ArcOutputNode()

    env_shape = ShapeSizeEnvOperator()
    graph.add_edge(output_object, env_shape)
    graph.add_edge(env_shape, environment.output_matrix_node)

    # Relationships
    rank = SetRankOperator()
    graph.add_edge(object_set, rank, from_port="*.identity.shape.count", to_port=0)
    graph.add_edge(object_node, rank, from_port="identity.shape.count", to_port=1)

    equals = Equals()
    graph.add_edge(rank, equals, to_port=0)
    graph.add_edge(zero, equals, to_port=1)

    # Operators
    relocate = RelocateOperator()
    graph.add_edge(object_node, relocate, from_port="identity", to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, output_object)

    return graph


# 50
def solve_253bf280():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    green = Constant(ColorValue, ColorValue(4))
    z_index_dot_solo = Constant(int, 1)

    # Input objects
    dot_solo = ArcInputNode(adjacency)
    dot_1 = ArcInputNode(adjacency)
    dot_2 = ArcInputNode(adjacency)

    # Output objects
    dot_out = ArcOutputNode()
    line = ArcOutputNode()

    # Matrix shape
    pass

    # Relationships
    aligned_cardinal = AlignedCardinal()
    graph.add_edge(dot_1, aligned_cardinal, to_port=0)
    graph.add_edge(dot_2, aligned_cardinal, to_port=1)

    # Operators
    persist = PersistOperator()
    graph.add_edge(dot_solo, persist)

    set_z_index_dot_solo = SetZIndexOperator()
    graph.add_edge(persist, set_z_index_dot_solo, to_port=0)
    graph.add_edge(z_index_dot_solo, set_z_index_dot_solo, to_port=1)
    graph.add_edge(set_z_index_dot_solo, dot_out)

    draw_line = DrawLineOperator()
    graph.add_edge(green, draw_line, to_port=0)
    graph.add_edge(dot_1, draw_line, from_port="location", to_port=1)
    graph.add_edge(dot_2, draw_line, from_port="location", to_port=2)
    graph.add_edge(draw_line, line)

    return graph


# 52
def solve_25d8a9c8():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    grey = Constant(ColorValue, ColorValue(6))
    bar = Constant(ArcObject, ArcObject([[1, 1, 1]]))

    # Input nodes
    objects = ArcInputNode(adjacency)

    # Output nodes
    grey_object = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals = Equals()
    graph.add_edge(objects, equals, from_port="identity.shape", to_port=0)
    graph.add_edge(bar, equals, from_port="identity.shape", to_port=1)

    # Operators
    recolor = RecolorOperator()
    graph.add_edge(objects, recolor, to_port=0)
    graph.add_edge(grey, recolor, to_port=1)
    graph.add_edge(recolor, grey_object)

    return graph


# 61
# NOTE SAME EXACT SOLUTION AS 17 - PATTERN INFILL
def solve_29ec7d0e():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    center = Constant(LocationValue, CENTERED)

    # Input nodes
    all_objects = InputSetNode(ArcObject, perception_qualifiers={adjacency})

    # Output nodes
    main_output_node = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    cluster_node = CreateCompositeObjectOperator()
    graph.add_edge(all_objects, cluster_node)

    pattern = GetClusterColorPatternOperator()
    graph.add_edge(cluster_node, pattern)

    # Operators
    create_rectangle = CreateRectangleOperator()
    graph.add_edge(pattern, create_rectangle, to_port=0)
    graph.add_edge(environment.input_matrix_node(), create_rectangle, from_port="x_size", to_port=1)
    graph.add_edge(environment.input_matrix_node(), create_rectangle, from_port="y_size", to_port=2)

    relocate = RelocateOperator()
    graph.add_edge(create_rectangle, relocate, to_port=0)
    graph.add_edge(center, relocate, to_port=1)
    graph.add_edge(relocate, main_output_node)

    return graph


# 143
def solve_63613498():
    environment = ArcEnvironment([adjacency], 1)
    graph = ArcGraph(environment)

    # Constants
    legend_shape = Constant(ShapeValue, ShapeValue.from_zoomed_mask(np.array([[0, 0, 0, 1],
                                                                              [0, 0, 0, 1],
                                                                              [0, 0, 0, 1],
                                                                              [1, 1, 1, 1],])))

    # Input nodes
    legend = ArcInputNode(adjacency)
    subject = ArcInputNode(adjacency)
    selected = ArcInputNode(adjacency)

    all_non_selected = ArcInputNode(adjacency)

    # Disjoint set
    disjoint_set_selected = DisjointSetNode(ArcObject)
    graph.add_edge(selected, disjoint_set_selected, to_port=0)

    # Output nodes
    recolored_node = ArcOutputNode()
    persisted_node = ArcOutputNode()

    # Matrix size
    pass

    # Relationships
    equals_legend_shape = Equals()
    graph.add_edge(legend, equals_legend_shape, to_port=0, from_port="identity.shape")
    graph.add_edge(legend_shape, equals_legend_shape, to_port=1)

    contains = ObjectContains()
    graph.add_edge(legend, contains, to_port=0)
    graph.add_edge(subject, contains, to_port=1)

    equals_shape = Equals()
    graph.add_edge(subject, equals_shape, to_port=0, from_port="identity.shape")
    graph.add_edge(selected, equals_shape, to_port=1, from_port="identity.shape")

    set_not_contains = SetNotContains()
    graph.add_edge(disjoint_set_selected, set_not_contains, to_port=0)
    graph.add_edge(all_non_selected, set_not_contains, to_port=1)

    # Operators
    recolor = RecolorOperator()
    graph.add_edge(selected, recolor, to_port=0)
    graph.add_edge(legend, recolor, to_port=1, from_port="identity.color")
    graph.add_edge(recolor, recolored_node)

    graph.add_edge(all_non_selected, persisted_node)

    return graph


# GOOD CASE 484b58aa - patterns
# GOOD CASE 4c177718 - semantics


def solve_wrapper(interpreter: Interpreter, graph, input_matrix, output_matrix):
    in_mat = np.array(input_matrix) + 1
    sol = interpreter.solve(graph, in_mat) - 1
    equal = np.array_equal(sol, np.array(output_matrix))
    if not equal:
        print(sol)
    # else:
    #     print("Passed")
    #     print(sol)
    return equal


class SolverTests(unittest.TestCase):

    def test_training_challenges(self):

        challenges = list(read_challenges("data/arc-agi_training_challenges.json").values())
        solutions = read_solutions("data/arc-agi_training_solutions.json")

        for challenge in challenges:

            # if challenge.challenge_id != "10fcaaa3":
            #     continue

            solver_function = globals().get("solve_" + challenge.challenge_id)
            if solver_function:
                reusable_graph = solver_function()
                reusable_graph.lazy_create_default_env_shape_program()
                interpreter = ArcInterpreter()

                with self.subTest(challenge=challenge.challenge_id):
                    passing = True
                    for case_idx, case in enumerate(challenge.train):

                        if not solve_wrapper(interpreter, reusable_graph, case.input_matrix, case.output_matrix):
                            passing = False

                    challenge_solutions = solutions[challenge.challenge_id]
                    for case_idx, case in enumerate(challenge.test):
                        if not solve_wrapper(interpreter, reusable_graph, case.input_matrix, challenge_solutions[case_idx]):
                            passing = False

                    self.assertTrue(passing)

if __name__ == "__main__":
    unittest.main()
