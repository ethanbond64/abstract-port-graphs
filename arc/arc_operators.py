import math
from copy import deepcopy, copy
from typing import Tuple, Union, Dict, Type

import numpy as np

from nodes import OperatorNode, Constant, InputNode, OutputNode
from port_graphs import PortGraph
from arc.arc_objects import (ArcObject, ColorValue, LocationValue, Direction, ObjectIdentity,
                                 ShapeValue,
                                 OneDimensionalLocationRelationship, Side, ConcreteLinearPattern,
                                 LinearLocation, ConcreteIdentityRelationship, LocationSchema,
                                 LocationSpecificAttribute, TwoDimensionalLocationRelationship,
                                 EMPTY_SHAPE,
                                 EMPTY_OBJECT, EnvironmentShape, ArcZone, apply_orientation,
                                 apply_orientation_vector, Orientation, CompositeArcObject)
from base_types import DslSet
from arc.arc_perception import all_cavities, outer_cavities
from arc.arc_utils import get_crow_distance_explicit, compress_matrix, expand_matrix, get_crow_distance, rotate_90, rotate_90_reverse
from operator_primitives import ConstructorOperator, MultiplyOperator
from functional_operator import FunctionalOperator


# Not-bijective
# TODO reconcile with the above operator, lots of redundancy with divergent types here
class ApplyTwoDimensionalRelationshipOperator(OperatorNode):

    def __init__(self):
        super().__init__([TwoDimensionalLocationRelationship, ArcObject], LocationValue)

    def apply(self, rel: TwoDimensionalLocationRelationship, obj: ArcObject):
        return self.create_location_from_relationship((rel.x_location_relationship, rel.y_location_relationship), obj)


    @staticmethod
    def create_location_from_relationship(
            relationship: Tuple[OneDimensionalLocationRelationship, OneDimensionalLocationRelationship],
            keystone_object: ArcObject) -> LocationValue:

        new_location = LocationValue(LinearLocation(), LinearLocation())
        new_location.x_location.env_shape = copy(keystone_object.location.x_location.env_shape)
        new_location.x_location.linear_env = keystone_object.location.x_location.linear_env
        new_location.y_location.env_shape = copy(keystone_object.location.y_location.env_shape)
        new_location.y_location.linear_env = keystone_object.location.y_location.linear_env

        for one_dim_relationship in relationship:
            dimension = one_dim_relationship.dimension
            new_dimension_location = new_location.get_dimension_location(dimension)

            # TODO uptake location schema
            if one_dim_relationship.aligned:
                keystone_value = keystone_object.location.get_dimension_location(dimension).center
                new_dimension_location.center = keystone_value
            elif one_dim_relationship.location_schema == LocationSchema.CLOSE_SIDES and one_dim_relationship.side == Side.SIDE_MIN:
                keystone_value = keystone_object.location.get_dimension_location(dimension).actual_min
                new_dimension_location.actual_max = keystone_value + one_dim_relationship.distance
            elif one_dim_relationship.location_schema == LocationSchema.CLOSE_SIDES and one_dim_relationship.side == Side.SIDE_MAX:
                keystone_value = keystone_object.location.get_dimension_location(dimension).actual_max
                new_dimension_location.actual_min = keystone_value - one_dim_relationship.distance
            elif one_dim_relationship.location_schema == LocationSchema.CENTERS and one_dim_relationship.side is None:
                keystone_dim_location = keystone_object.location.get_dimension_location(dimension)
                keystone_value = keystone_dim_location.center
                # TODO confirm you can use this env??!!?
                keystone_env = keystone_dim_location.linear_env
                context_correct_delta = one_dim_relationship.distance / keystone_env
                new_dimension_location.center = keystone_value + context_correct_delta
            else:
                raise Exception("Invalid side")

        return new_location


# Not-bijective, but very nicely limited arg space...
class ApplyIdentityRelationshipOperator(OperatorNode):

    def __init__(self):
        super().__init__([ConcreteIdentityRelationship, ObjectIdentity], ObjectIdentity)

    # TODO make this work for non-uniform colors too
    def apply(self, relationship: ConcreteIdentityRelationship, object_identity: ObjectIdentity) -> ObjectIdentity:

        norm_mask = object_identity.zoomed_mask
        original_scale = object_identity.shape.scale
        if original_scale > 1:
            norm_mask = compress_matrix(norm_mask, original_scale, 0)

        applied_mask = apply_orientation(norm_mask, relationship.delta_orientation)

        target_scale = original_scale * relationship.scale_factor
        if math.isclose(target_scale, round(target_scale), abs_tol=0.0001):
            target_scale = int(target_scale)
        else:
            raise Exception("Invalid scale application")

        if target_scale > 1:
            applied_mask = expand_matrix(applied_mask, target_scale, 0)

        new_identity = ObjectIdentity.from_zoomed_mask(np.array(applied_mask))
        return new_identity


# Not-bijective
# Given an identity transform relationship, apply that transform to the location relationship
class TransformLocationRelationshipOperator(OperatorNode):
    def __init__(self):
        super().__init__([ConcreteIdentityRelationship, TwoDimensionalLocationRelationship],
                         TwoDimensionalLocationRelationship)

    def apply(self, identity_relationship: ConcreteIdentityRelationship,
              location_relationship: TwoDimensionalLocationRelationship) -> TwoDimensionalLocationRelationship:

        coords = (copy(location_relationship.x_location_relationship), copy(location_relationship.y_location_relationship))
        norm_x, norm_y = apply_orientation_vector(*coords, identity_relationship.delta_orientation)
        norm_x.distance = norm_x.distance * identity_relationship.scale_factor
        norm_y.distance = norm_y.distance * identity_relationship.scale_factor
        new_location_relationship = TwoDimensionalLocationRelationship(norm_x, norm_y)

        return new_location_relationship


class TranslateIdentityOperator(OperatorNode):
    def __init__(self):
        super().__init__([ObjectIdentity, Orientation], ObjectIdentity)

    def apply(self, object_identity: ObjectIdentity, orientation: Orientation) -> ObjectIdentity:
        new_zoomed_mask = apply_orientation(object_identity.zoomed_mask, orientation)
        new_identity = ObjectIdentity.from_zoomed_mask(new_zoomed_mask)
        return new_identity


# TODO uncomment when ready
# class MeasureIdentityTranslationDifference(OperatorNode):
#     def __init__(self):
#         super().__init__([ObjectIdentity, ObjectIdentity], Orientation)
#
#     def apply(self, object_identity_1: ObjectIdentity, object_identity_2: ObjectIdentity) -> Orientation:
#         if object_identity_1.norm_image.array_hash != object_identity_2.norm_image.array_hash:
#             raise ValueError("Cannot measure difference between two different identities")
#         orientation = difference_orientation(object_identity_1.orientation, object_identity_2.orientation)
#         return orientation


# Not-bijective
class ApplyLinearPatternOperator(OperatorNode):

    def __init__(self):
        super().__init__([ConcreteLinearPattern, int], ArcObject)

    def apply(self, pattern: ConcreteLinearPattern, distance: Union[int, float]) -> ArcObject:

        if pattern.direction == Direction.DIRECTION_X:
            raise NotImplementedError("X direction not supported yet")

        vector_list = []
        for i in range(int(distance)):
            index = i % len(pattern.sub_objects)
            sub_object = pattern.sub_objects[index]
            vector_list.append(sub_object.mask)

        new_mask = np.vstack(vector_list)
        return ArcObject(new_mask)


# Bijective
class CreateRectangleOperator(OperatorNode):
    def __init__(self):
        super().__init__([ColorValue, int, int], ObjectIdentity)

    def apply(self, color_value: ColorValue, width: int, height: int) -> ObjectIdentity:
        mask = np.ones((height, width))
        shape = ShapeValue.from_zoomed_mask(mask)
        identity = ObjectIdentity(color_value, shape)
        identity.rebuild_zoomed_mask()
        return identity


# Bijective
class DrawLineOperator(OperatorNode):
    def __init__(self):
        super().__init__([ColorValue, LocationValue, LocationValue], ArcObject)

    def apply(self, color_value: ColorValue, start_location: LocationValue, end_location: LocationValue) -> ArcObject:
        # TODO maybe need location schema as an arg - currently always using the center of the start/end locations

        # Assert all env sizes are the same
        if not (start_location.x_location.env_shape == end_location.x_location.env_shape == start_location.y_location.env_shape == end_location.y_location.env_shape):
            raise Exception("Invalid args - different environment sizes")

        env_x = start_location.x_location.env_shape.x_size
        env_y = start_location.y_location.env_shape.y_size

        start_x = int(start_location.x_location.center * env_x)
        start_y = int(start_location.y_location.center * env_y)
        end_x = int(end_location.x_location.center * env_x)
        end_y = int(end_location.y_location.center * env_y)

        delta_x = end_x - start_x
        delta_y = end_y - start_y

        if delta_x == delta_y == 0:
            raise Exception("Invalid args - same location")

        if not(delta_x == 0 or delta_y == 0 or abs(delta_x) == abs(delta_y)):
            raise Exception("Invalid args - not aligned vertically, horizontally, or diagonally")

        current_x = start_x
        current_y = start_y
        steps = max(abs(delta_x), abs(delta_y))
        mask = np.zeros(start_location.x_location.env_shape.to_tuple())
        for _ in range(steps + 1):
            mask[current_y, current_x] = color_value.value

            if delta_x < 0:
                current_x = current_x - 1
            elif delta_x > 0:
                current_x = current_x + 1

            if delta_y < 0:
                current_y = current_y - 1
            elif delta_y > 0:
                current_y = current_y + 1

        return ArcObject(mask)


# Bijective (injective only)
class DivideIdentityOperator(OperatorNode):
    def __init__(self):
        super().__init__([ObjectIdentity, int, int], DslSet[ObjectIdentity])

    def apply(self, input_identity: ObjectIdentity, x_divisions: int, y_divisions: int) -> DslSet[ObjectIdentity]:
        mask = input_identity.zoomed_mask
        out_identities = DslSet()

        x_division_size = mask.shape[1] / x_divisions
        y_division_size = mask.shape[0] / y_divisions

        # TODO what is actual desired behavior?
        if x_division_size % 1 != 0 or y_division_size % 1 != 0:
            raise Exception("Invalid division size")

        x_division_size = int(x_division_size)
        y_division_size = int(y_division_size)

        for x_division in range(x_divisions):
            for y_division in range(y_divisions):
                x_start = x_division * x_division_size
                x_end = x_start + x_division_size
                y_start = y_division * y_division_size
                y_end = y_start + y_division_size
                new_mask = mask[y_start:y_end, x_start:x_end]
                new_identity = ObjectIdentity.from_zoomed_mask(new_mask, zone=True)
                out_identities.append(new_identity)

        return out_identities


# TODO REMOVE
class SpecificLocationValueOperator(OperatorNode):
    def __init__(self):
        super().__init__([LinearLocation, LocationSpecificAttribute], LinearLocation)

    def apply(self, location: LinearLocation, attribute: LocationSpecificAttribute) -> LinearLocation:

        new_linear_location = deepcopy(location)
        new_linear_location.center = location.center if attribute == LocationSpecificAttribute.CENTER else None
        new_linear_location.relative_min = location.relative_min if attribute == LocationSpecificAttribute.RELATIVE_MIN else None
        new_linear_location.relative_max = location.relative_max if attribute == LocationSpecificAttribute.RELATIVE_MAX else None
        new_linear_location.actual_min = location.actual_min if attribute == LocationSpecificAttribute.ACTUAL_MIN else None
        new_linear_location.actual_max = location.actual_max if attribute == LocationSpecificAttribute.ACTUAL_MAX else None

        return new_linear_location


class FindClosestOperator(OperatorNode):
    def __init__(self):
        super().__init__([DslSet[ArcObject], ArcObject], ArcObject)

    def apply(self, object_set: DslSet[ArcObject], target_object: ArcObject) -> ArcObject:

        target_x = target_object.location.x_location.center
        target_y = target_object.location.y_location.center

        # TODO currently only uses center, need to support location schema
        best = None
        lowest_distance = float('inf')
        for obj in object_set:
            obj_x = obj.location.x_location.center
            obj_y = obj.location.y_location.center
            distance = get_crow_distance_explicit(obj_x, obj_y, target_x, target_y)
            if distance < lowest_distance:
                lowest_distance = distance
                best = obj

        return best


# Not-bijective
class MeasureCentersOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject, ArcObject], TwoDimensionalLocationRelationship)

    def apply(self, obj_1: ArcObject, obj_2: ArcObject) -> TwoDimensionalLocationRelationship:
        # TODO concrete...should call into this ^ apply method, not the other way around
        return MeasureCentersOperator.apply_with_second_arg_location(obj_1, obj_2.location)

    @staticmethod
    def apply_with_second_arg_location(obj_1: ArcObject, loc_2: LocationValue):
        x_rel = OneDimensionalLocationRelationship.get_one_dimensional_relationship(obj_1.location, loc_2,
                                                                              Direction.DIRECTION_X,
                                                                              LocationSchema.CENTERS)
        y_rel = OneDimensionalLocationRelationship.get_one_dimensional_relationship(obj_1.location, loc_2,
                                                                              Direction.DIRECTION_Y,
                                                                              LocationSchema.CENTERS)
        return TwoDimensionalLocationRelationship(x_rel, y_rel)


# Not-bijective
class MeasureClosestSidesOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject, ArcObject], TwoDimensionalLocationRelationship)

    def apply(self, obj_1: ArcObject, obj_2: ArcObject) -> TwoDimensionalLocationRelationship:
        # TODO concrete...should call into this ^ apply method, not the other way around
        return MeasureClosestSidesOperator.apply_with_second_arg_location(obj_1, obj_2.location)

    @staticmethod
    def apply_with_second_arg_location(obj_1: ArcObject, loc_2: LocationValue):
        x_rel = OneDimensionalLocationRelationship.get_one_dimensional_relationship(obj_1.location, loc_2,
                                                                              Direction.DIRECTION_X,
                                                                              LocationSchema.CLOSE_SIDES)
        y_rel = OneDimensionalLocationRelationship.get_one_dimensional_relationship(obj_1.location, loc_2,
                                                                              Direction.DIRECTION_Y,
                                                                              LocationSchema.CLOSE_SIDES)
        return TwoDimensionalLocationRelationship(x_rel, y_rel)


# Not-bijective
# TODO Nore there can be several valid relationships -
#  this method is dangerous because it only returns the simplest one. Could cause a loss of information.
class MeasureIdentityDeltaOperator(OperatorNode):
    def __init__(self):
        super().__init__([ObjectIdentity, ObjectIdentity], ConcreteIdentityRelationship)

    def apply(self, object_identity_1: ObjectIdentity, object_identity_2: ObjectIdentity) -> ConcreteIdentityRelationship:
        relationship = ConcreteIdentityRelationship(object_identity_1, object_identity_2)
        return relationship


class SumLocationRelationshipOperator(OperatorNode):
    def __init__(self):
        super().__init__([TwoDimensionalLocationRelationship, TwoDimensionalLocationRelationship], TwoDimensionalLocationRelationship)

    def apply(self, rel_1: TwoDimensionalLocationRelationship, rel_2: TwoDimensionalLocationRelationship) -> TwoDimensionalLocationRelationship:

        if rel_1.location_schema != rel_2.location_schema != LocationSchema.CENTERS:
            raise Exception("rel_1 and rel_2 must have same schema - centers")

        x_sum = rel_1.x_location_relationship.distance + rel_2.x_location_relationship.distance
        new_x_rel = OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_X, None, x_sum, x_sum == 0) # TODO hard coded

        y_sum = rel_1.y_location_relationship.distance + rel_2.y_location_relationship.distance
        new_y_rel = OneDimensionalLocationRelationship(LocationSchema.CENTERS, Direction.DIRECTION_Y, None, y_sum, y_sum == 0)

        return TwoDimensionalLocationRelationship(new_x_rel, new_y_rel)


# Given an object and a shape, find all the places you could "cut" the shape from the object
class PossibleCutLocationsOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject, ObjectIdentity], DslSet[LocationValue])

    def apply(self, larger_object: ArcObject, cutter_shape: ObjectIdentity) -> DslSet[LocationValue]:

        if larger_object is EMPTY_OBJECT:
            return DslSet()

        matrix1 = larger_object.mask
        matrix2 = cutter_shape.zoomed_mask
        rows1, cols1 = matrix1.shape
        rows2, cols2 = matrix2.shape

        if rows2 >= rows1 or cols2 >= cols1:
            raise ValueError("Second matrix must be strictly smaller than the first matrix.")

        temp_objects = []

        # Iterate over every possible top-left coordinate for placing matrix2 in matrix1.
        for i in range(rows1 - rows2 + 1):
            for j in range(cols1 - cols2 + 1):
                # Extract the submatrix of matrix1 for the current position.
                window = matrix1[i:i + rows2, j:j + cols2]

                # Check the condition: the pattern of nonzeros must match.
                if np.array_equal(window != 0, matrix2 != 0):
                    # Create a new matrix of the same shape as matrix1 with zeros.
                    new_matrix = np.zeros_like(matrix1)
                    # Embed matrix2 into the new matrix at position (i, j).
                    new_matrix[i:i + rows2, j:j + cols2] = matrix2
                    temp_objects.append(ArcObject(new_matrix))

        return DslSet(o.location for o in temp_objects)


# Given an object, shape, and location = cut the shape from the object at the location (absolute).
# Return a new arcobject representing the original object, but with the shape cut out of it.
class CutShapeOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject, ShapeValue, LocationValue], ArcObject)

    def apply(self, original_object: ArcObject, cutter_shape: ShapeValue, cutter_location: LocationValue) -> ArcObject:

        if cutter_shape == EMPTY_SHAPE:
            return copy(original_object)

        temp_identity = ObjectIdentity(ColorValue(1), copy(cutter_shape))
        temp_identity.rebuild_zoomed_mask()
        temp_object = ArcObject(None)
        temp_object.identity = temp_identity
        temp_object.location = copy(cutter_location)

        new_mask = copy(original_object.mask)
        new_mask[temp_object.mask == 1] = 0
        if not np.any(new_mask):
            new_object = EMPTY_OBJECT
        else:
            new_object = ArcObject(new_mask)

        return new_object


class DivideToZonesOperator(OperatorNode):
    def __init__(self):
        super().__init__([EnvironmentShape, int, int], DslSet[ArcZone])

    def apply(self, env_shape: EnvironmentShape, x_divisor: int, y_divisor: int) -> DslSet[ArcZone]:

        x_unit = int(env_shape.x_size / x_divisor)
        y_unit = int(env_shape.y_size / y_divisor)

        zone_shape_img = np.ones((y_unit, x_unit))

        all_zones = DslSet()
        cnt = 0
        for y_index in range(y_divisor):
            for x_index in range(x_divisor):
                x_min = x_index * x_unit
                x_max = x_min + x_unit
                y_min = y_index * y_unit
                y_max = y_min + y_unit
                x_location = LinearLocation(actual_min=x_min, actual_max=x_max, linear_env=env_shape.x_size, env_shape=copy(env_shape))
                y_location = LinearLocation(actual_min=y_min, actual_max=y_max, linear_env=env_shape.y_size, env_shape=copy(env_shape))
                location = LocationValue(x_location, y_location)
                zone = ArcZone(zone_shape_img.copy(), location, cnt)
                all_zones.append(zone)
                cnt += 1

        return all_zones


# TODO NORM - make this work for "fences" that are lines, where the cavity is between it and the boundary
# TODO NORM - make this work as a non-primitive which uses a primitive cavities operator to get the cavities.
class GetCavityZonesOperator(OperatorNode):
    def __init__(self):
        super().__init__([DslSet[ArcObject]], DslSet[ArcZone])

    def apply(self, obj_in: ArcObject) -> DslSet[ArcZone]:
        objects = all_cavities(obj_in.mask)

        loc_to_zone: Dict[int, ArcZone] = {}
        for obj in objects:
            zone_img = np.ones((obj.identity.shape.bounding_height, obj.identity.shape.bounding_width))
            zone = ArcZone(zone_img, copy(obj.location), -1)
            computed_loc = (zone.location.x_location.actual_min * zone.location.y_location.actual_min) + zone.location.x_location.actual_min
            if computed_loc in loc_to_zone:
                raise RuntimeError("Duplicate zone location")
            loc_to_zone[computed_loc] = zone

        return DslSet(map(sorted(loc_to_zone, key = lambda tup: tup[0]), lambda tup: tup[1]))


class GetOuterCavitiesOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject], DslSet[ArcObject])

    def apply(self, obj: ArcObject) -> DslSet[ArcObject]:
        return DslSet(outer_cavities(obj.mask))


class GetZoneByIndexOperator(OperatorNode):
    def __init__(self):
        super().__init__([DslSet[ArcZone], int], ArcZone)

    def apply(self, zone_list: DslSet[ArcZone], zone_index: int) -> ArcZone:
        return zone_list[zone_index]


# TODO cleanup to
class ZoneLocationToFullLocationOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcZone, LocationValue], LocationValue)

    def apply(self, zone: ArcZone, object_location: LocationValue) -> LocationValue:

        new_x_location = LinearLocation()
        new_x_location.center = None
        new_x_location.relative_min = None
        new_x_location.relative_max = None
        new_x_location.actual_min = object_location.x_location.actual_min + zone.location.x_location.actual_min
        # new_x_location.actual_max = object_location.x_location.actual_max + zone.location.x_location.actual_min
        new_x_location.env_shape = zone.location.x_location.env_shape
        new_x_location.linear_env = zone.location.x_location.linear_env

        new_y_location = LinearLocation()
        new_y_location.center = None
        new_y_location.relative_min = None
        new_y_location.relative_max = None
        new_y_location.actual_min = object_location.y_location.actual_min + zone.location.y_location.actual_min
        # new_y_location.actual_max = object_location.y_location.actual_max + zone.location.y_location.actual_min
        new_y_location.env_shape = zone.location.y_location.env_shape
        new_y_location.linear_env = zone.location.y_location.linear_env

        return LocationValue(new_x_location, new_y_location)


# TODO need some construct other than color as output type... ideally a normalized cluster.
class GetClusterColorPatternOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject], ColorValue)

    def apply(self, arc_object: ArcObject) -> ColorValue:

        # Short circuit for consistent colors
        if arc_object.identity.color.value % 1 == 0:
            return arc_object.identity.color

        # TODO - to handle all cases, where the shape has many wholes and no "full" pattern rectangles,
        #  is very difficult. Just try to make it work for 17.
        driver_mask = copy(arc_object.identity.zoomed_mask)  # TODO Confirm correct copy function

        # Start with finding repetition in 1D along the top row
        # TODO take multiple rows' cuts and use the largest (must be multiple of the others)
        # TODO temp - taking the max of the top 3
        cut_x_1 = get_one_dimensional_matching_cut(driver_mask[0])
        cut_x_2 = get_one_dimensional_matching_cut(driver_mask[1])
        cut_x_3 = get_one_dimensional_matching_cut(driver_mask[2])
        cut_x_4 = get_one_dimensional_matching_cut(driver_mask[3])
        cut_x_5 = get_one_dimensional_matching_cut(driver_mask[4])
        cut_x = max(cut_x_1, cut_x_2, cut_x_3, cut_x_4, cut_x_5)

        # Move to first column once found
        # TODO temp - shortsighted need to handle all sizes
        cut_y_1 = get_one_dimensional_matching_cut(driver_mask[:, 0])
        cut_y_2 = get_one_dimensional_matching_cut(driver_mask[:, 1])
        cut_y_3 = get_one_dimensional_matching_cut(driver_mask[:, 2])
        cut_y_4 = get_one_dimensional_matching_cut(driver_mask[:, 3])
        cut_y_5 = get_one_dimensional_matching_cut(driver_mask[:, 4])
        cut_y = max(cut_y_1, cut_y_2, cut_y_3, cut_y_4, cut_y_5)

        # TODO NAIVE create the normalized patch using first row and column.
        # Also fill in the gaps in the patch
        patch = driver_mask[:cut_y, :cut_x]
        x_cuts = len(driver_mask[0]) // cut_x
        y_cuts = len(driver_mask[:, 0]) // cut_y
        break_loops = False
        for x in range(x_cuts):
            for y in range(y_cuts):
                if not np.any(patch == 0):
                    break_loops = True
                    break

                first_x = cut_x * x
                first_y = cut_y * y
                other_patch = driver_mask[first_x:first_x + cut_x, first_y:first_y + cut_y]

                # TODO bad edge case behavior - should instead shrink the matrix and compare the smaller section instead of continuing
                if patch.shape != other_patch.shape:
                    continue

                # Try equality first (throw if not equal)
                if not test_sub_matrix_equality(patch, other_patch):
                    print("UNIMPLEMENTED COLOR PATTERN PATCH BAD VALUE")
                    raise Exception()

                patch[(patch == 0) & (other_patch != 0)] = other_patch[(patch == 0) & (other_patch != 0)]

            if break_loops:
                break

        return ColorValue.from_pattern_patch(patch)


class GetClusterRadialColorPatternOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject], ColorValue)

    def apply(self, arc_object: ArcObject) -> ColorValue:

        # TODO radial currently enforces that ALL squares of a certain crow-distance are the same color
        # TODO need more controls to specify where the center is and also other operators to get the center's location.
        matrix = arc_object.identity.zoomed_mask
        rows, cols = matrix.shape
        center = (rows // 2, cols // 2)
        distance_dict = {}

        for r in range(rows):
            for c in range(cols):
                if matrix[r, c] != 0:
                    distance = get_crow_distance(center, r, c)
                    if distance in distance_dict and distance_dict[distance] != matrix[r, c]:
                        raise ValueError(
                            f"Conflicting values at distance {distance}: {distance_dict[distance]} vs {matrix[r, c]}")
                    distance_dict[distance] = matrix[r, c]

        return ColorValue.from_radial_dict(distance_dict)


# TODO NORM can we get rid of the concrete linear pattern operator?
class GetLinearShapePatternOperator(OperatorNode):

    def __init__(self):
        super().__init__([ArcObject, Direction], ConcreteLinearPattern)

    def apply(self, arc_object: ArcObject, direction: Direction) -> ConcreteLinearPattern:
        # split into rows or columns based on the direction
        sub_objects = []
        if direction == Direction.DIRECTION_Y:
            for row in arc_object.mask:
                sub_objects.append(ArcObject(np.array([row]), None))
        else:
            raise NotImplementedError("Test that the T transpose below works! This is untested and might not work")
            # for col in arc_object.mask.T:
            #     sub_objects.append(ArcObject(col, None))

        def object_equals(obj1, obj2):
            return np.array_equal(obj1.mask, obj2.mask)

        # Try all (n-1) possible pattern cut locations
        # case cut == 1 -> verify all sub_objects are equal
        if len(sub_objects) == 1 or all([object_equals(sub_objects[0], obj) for obj in sub_objects[1:]]):
            return ConcreteLinearPattern([sub_objects[0]], direction)

        # case no cut -> the entire object is the pattern
        if len(sub_objects) == 2 or not any([object_equals(sub_objects[0], obj) for obj in sub_objects[1:]]):
            return ConcreteLinearPattern(sub_objects, direction)

        # Try cuts from 2 to n-1
        all_indices = list(range(len(sub_objects)))
        for cut in range(2, len(sub_objects)):
            success = True
            for expected_modulo in range(cut):
                current_indices = filter(lambda i: i % cut == expected_modulo, all_indices)
                current_objects = [sub_objects[i] for i in current_indices]
                if not all([object_equals(current_objects[0], obj) for obj in current_objects[1:]]):
                    success = False
                    break
            if success:
                return ConcreteLinearPattern(sub_objects[:cut], direction)

        # Cases cut == 2
        # index % cut  == 0, index % cut == 1
        # 0 == 2 == 4 == 6... and 1 == 3 == 5 == 7...,

        # Cases cut == 3
        # index % cut  == 0, index % cut == 1, index % cut == 2
        # 0 == 3 == 6 == 9... and 1 == 4 == 7 == 10... and 2 == 5 == 8 == 11...

        # Base case (no matches, return full sub_objects)
        return ConcreteLinearPattern(sub_objects, direction)


# 0 is wildcard on both sides (matching vs matched)
def test_sub_matrix_equality(sub1, sub2):
    # If different shapes, throw
    if sub1.shape != sub2.shape:
        print("Invalid submatrix equality args")
        raise Exception("Invalid submatrix equality args")

    # Copy all values from sub 1 to working matrix
    working_matrix_1 = np.copy(sub1)
    working_matrix_2 = np.copy(sub2)
    # Set any cells that are 0 in sub 2 to 0
    working_matrix_1[working_matrix_2 == 0] = 0
    working_matrix_2[working_matrix_1 == 0] = 0
    # Subtract sub 2
    working_matrix = working_matrix_1 - working_matrix_2
    # return boolean == all cells are 0
    return not np.any(working_matrix)


# Gets the cut size required for repetition in the 1D array. If not are found, it returns the full length of the array.
def get_one_dimensional_matching_cut(arr):
    passing = False
    cut = 1
    while not passing:
        # Try cut, brute force
        total_cuts = len(arr) // cut

        continue_while = False
        for cut_index in range(total_cuts):
            first_start = cut_index * cut
            first = arr[first_start:first_start + cut]
            second = arr[first_start + cut:first_start + cut + cut]

            if len(second) < len(first):
                first = first[:len(second)]

            if not test_sub_matrix_equality(first, second):
                continue_while = True
                break

        if continue_while:
            cut += 1
            continue

        passing = True
        break

    if passing:
        return cut

    return len(arr)


# TODO NORM - use a single primitive with an operator enum
# NEED TO THINK ABOUT COMPOSITE TRANSFORM DATA TYPE

class Rotate90Operator(OperatorNode):
    def __init__(self):
        super().__init__([ObjectIdentity, int], ObjectIdentity)

    def apply(self, identity: ObjectIdentity, revolutions: int) -> ObjectIdentity:
        if revolutions == 0:
            return identity

        rotation_fn = rotate_90 if revolutions > 0 else rotate_90_reverse

        mask = identity.zoomed_mask

        for _rev in range(abs(revolutions)):
            mask = rotation_fn(mask)

        return ObjectIdentity.from_zoomed_mask(mask) # TODO handle zone!


# TODO NORM - use a single primitive with an operator enum - make the types shape?? OR object ??
#  Location seems important - even if its relative

# Apply binary AND operation to two object identities wherever a non-zero value is present in both objects
# The remaining values are set to the color from the first object in the environment shape of the first object.
class BinaryAndOperator(OperatorNode):

    def __init__(self):
        super().__init__([ObjectIdentity, ObjectIdentity], ObjectIdentity)

    def apply(self, input_identity_1: ObjectIdentity, input_identity_2: ObjectIdentity) -> ObjectIdentity:
        new_mask = input_identity_1.zoomed_mask * (input_identity_2.zoomed_mask != 0)
        new_object = ArcObject.create_zone_object(new_mask)
        return new_object.identity


class BinaryNorOperator(OperatorNode):

    def __init__(self):
        super().__init__([ObjectIdentity, ObjectIdentity], ObjectIdentity)

    def apply(self, input_identity_1: ObjectIdentity, input_identity_2: ObjectIdentity) -> ObjectIdentity:
        new_mask = (input_identity_1.zoomed_mask == 0) * (input_identity_2.zoomed_mask == 0)
        new_object = ArcObject.create_zone_object(new_mask)
        return new_object.identity


class MoveOperator(OperatorNode):
    def __init__(self):
        super().__init__([ArcObject, Direction, int], ArcObject)

    def apply(self, input_object: ArcObject, direction: Direction, distance: int) -> ArcObject:
        obj = deepcopy(input_object)
        location = obj.location.get_dimension_location(direction)
        location.center = None
        location.relative_min = None
        location.relative_max = None
        location.actual_min += distance
        location.actual_max += distance
        obj.location.set_dimension_location(direction, location)
        # obj.refresh_location_from_components(location.get_env())
        return obj


# Given an object to move and an object to move towards, find the location adjacent to the target object
# with the minimum movement from the original location
# TODO really this should return location value as a POINT on the target object (abs min/max)
class ApproachOperator(OperatorNode):

    def __init__(self):
        super().__init__([ArcObject, ArcObject], ArcObject)

    def apply(self, input_object: ArcObject, target_object: ArcObject) -> ArcObject:

        new_object = ArcObject(input_object.mask)

        for direction in [Direction.DIRECTION_X, Direction.DIRECTION_Y]:
            location = new_object.location.get_dimension_location(direction)
            location.center = None
            location.relative_min = None
            location.relative_max = None

            target_location = target_object.location.get_dimension_location(direction)

            if location.actual_max < target_location.actual_min:
                location.actual_max = target_location.actual_min
                location.actual_min = None
            elif location.actual_min > target_location.actual_max:
                location.actual_min = target_location.actual_max
                location.actual_max = None

            new_object.location.set_dimension_location(direction, location)
        return new_object


# Not-bijective
# Move a set of objects in a direction until they hit the edge or another object.
# TODO add primitives so that this operator can be written in pure dsl
# TODO need to be able to do this with different z indexes...
class CollapseDirectionOperator(OperatorNode):

    def __init__(self):
        super().__init__([DslSet[ArcObject], Direction, int], DslSet[ArcObject])

    def apply(self, objects: DslSet[ArcObject], dimension: Direction, sign: int) -> DslSet[ArcObject]:

        if dimension == Direction.DIRECTION_Y:
            axis = 0
        elif dimension == Direction.DIRECTION_X:
            axis = 1
        else:
            raise Exception("Invalid dimension")

        positive = sign >= 0

        # Assert all objects in the set have a mask of the same shape
        first_mask_shape = list(objects)[0].mask.shape
        if not all(first_mask_shape == o.mask.shape for o in objects):
            raise Exception("Invalid object set to collapse operator, all mask shapes must be equal")

        # Validate that no overlapping non-zero values already exist
        occupied = np.zeros(first_mask_shape, dtype=bool)
        matrices = []
        for obj in objects:
            mat = obj.mask
            matrices.append(copy(mat))
            if np.any(occupied & (mat != 0)):
                raise ValueError("Input matrices contain overlapping non-zero values.")
            occupied |= (mat != 0)

        moved = True
        while moved:
            moved = False
            new_occupied = np.zeros_like(occupied)

            for mat in matrices:
                new_mat = np.zeros_like(mat)

                bottom = -1 if positive else 0
                if axis == 0:
                    new_mat[bottom, :] = mat[bottom, :]
                else:
                    new_mat[:, bottom] = mat[:, bottom]

                # Determine iteration order based on direction
                if positive:
                    range_vals = range(first_mask_shape[axis] - 2, -1, -1)
                else:
                    range_vals = range(1, first_mask_shape[axis])

                # Iterate from bottom up to avoid conflicts in the same matrix
                for i in range_vals:
                    for j in range(first_mask_shape[1 - axis]):
                        index = (i, j) if axis == 0 else (j, i)
                        delta = 1 if positive else -1
                        next_index = (i + delta, j) if axis == 0 else (j, i + delta)
                        if mat[index] != 0 and 0 <= next_index[axis] < first_mask_shape[axis] and mat[next_index] == 0 and not \
                        occupied[next_index]:
                            new_mat[next_index] = mat[index]
                            moved = True
                        else:
                            new_mat[index] = mat[index]

                # Update the matrix in place
                mat[:] = new_mat
                new_occupied |= (new_mat != 0)

            occupied = new_occupied

        return DslSet(ArcObject(m) for m in matrices)


class PersistOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ArcObject], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, input_node: InputNode) -> PortGraph:
        graph.add_edge(input_node, output_node)
        return graph


class RecolorOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ArcObject, ColorValue], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, arc_object_node: InputNode,
                   color_node: InputNode) -> PortGraph:
        identity_type = Constant(Type, ObjectIdentity)
        object_type = Constant(Type, ArcObject.constructor_1)

        identity_constructor = ConstructorOperator()
        graph.add_edge(identity_type, identity_constructor, to_port=0)
        graph.add_edge(color_node, identity_constructor, to_port=1)
        graph.add_edge(arc_object_node, identity_constructor, from_port="identity.shape", to_port=2)

        object_constructor = ConstructorOperator()
        graph.add_edge(object_type, object_constructor, to_port=0)
        graph.add_edge(identity_constructor, object_constructor, to_port=1)
        graph.add_edge(arc_object_node, object_constructor, from_port="location", to_port=2)
        graph.add_edge(object_constructor, output_node)

        return graph


# TODO this is an exact copy of the object constructor
class RelocateOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ObjectIdentity, LocationValue], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, identity_node: InputNode,
                   location_node: InputNode) -> ArcObject:

        object_type = Constant(Type, ArcObject.constructor_1)

        object_constructor = ConstructorOperator()
        graph.add_edge(object_type, object_constructor, to_port=0)
        graph.add_edge(identity_node, object_constructor, to_port=1)
        graph.add_edge(location_node, object_constructor, to_port=2)
        graph.add_edge(object_constructor, output_node)

        return graph


class RelocateCardinalOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ObjectIdentity, LinearLocation, LinearLocation], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, identity_node: InputNode,
                   x_location: InputNode, y_location: InputNode) -> PortGraph:

        location_type = Constant(Type, LocationValue)

        location_constructor = ConstructorOperator()
        graph.add_edge(location_type, location_constructor, to_port=0)
        graph.add_edge(x_location, location_constructor, to_port=1)
        graph.add_edge(y_location, location_constructor, to_port=2)

        relocate = RelocateOperator()
        graph.add_edge(identity_node, relocate, to_port=0)
        graph.add_edge(location_constructor, relocate, to_port=1)
        graph.add_edge(relocate, output_node)

        return graph


# TODO temp until completions code understands constructors
class ReshapeOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ArcObject, ShapeValue], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, input_object: InputNode,
                   shape_node: InputNode) -> PortGraph:

        identity_type = Constant(Type, ObjectIdentity)

        identity_constructor = ConstructorOperator()
        graph.add_edge(identity_type, identity_constructor, to_port=0)
        graph.add_edge(input_object, identity_constructor, from_port="identity.color", to_port=1)
        graph.add_edge(shape_node, identity_constructor, to_port=2)

        relocate = RelocateOperator()
        graph.add_edge(identity_constructor, relocate, to_port=0)
        graph.add_edge(input_object, relocate, from_port="location", to_port=1)
        graph.add_edge(relocate, output_node)

        return graph


class CreateCompositeObjectOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([DslSet[ArcObject]], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, input_set_node: InputNode) -> PortGraph:
        composite_type = Constant(Type, CompositeArcObject)

        composite_constructor = ConstructorOperator()
        graph.add_edge(composite_type, composite_constructor, to_port=0)
        graph.add_edge(input_set_node, composite_constructor, to_port=1)
        graph.add_edge(composite_constructor, output_node)

        return graph


class SetZIndexOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ArcObject, int], ArcObject)

    def load_graph(self, graph: PortGraph, output_node: OutputNode,
                   object_node: InputNode, z_index_node: InputNode) -> PortGraph:

        object_type = Constant(Type, ArcObject.constructor_1) # Using default arg here - different from how synthesis works

        constructor = ConstructorOperator()
        graph.add_edge(object_type, constructor, to_port=0)
        graph.add_edge(object_node, constructor, from_port="identity", to_port=1)
        graph.add_edge(object_node, constructor, from_port="location", to_port=2)
        graph.add_edge(z_index_node, constructor, to_port=3)

        graph.add_edge(constructor, output_node)

        return graph


class ShapeSizeEnvOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([ArcObject], EnvironmentShape)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, input_object: InputNode) -> PortGraph:

        env_shape_type = Constant(Type, EnvironmentShape)

        env_shape_constructor = ConstructorOperator()
        graph.add_edge(env_shape_type, env_shape_constructor, to_port=0)
        graph.add_edge(input_object, env_shape_constructor, from_port="identity.shape.bounding_height", to_port=1)
        graph.add_edge(input_object, env_shape_constructor, from_port="identity.shape.bounding_width", to_port=2)
        graph.add_edge(env_shape_constructor, output_node)

        return graph


class ProportionalEnvOperator(FunctionalOperator):
    def __init__(self):
        super().__init__([EnvironmentShape, int, int], EnvironmentShape)

    def load_graph(self, graph: PortGraph, output_node: OutputNode, input_shape: InputNode,
                   x_proportion: InputNode, y_proportion: InputNode) -> PortGraph:
        env_shape_type = Constant(Type, EnvironmentShape)

        multiply_x = MultiplyOperator()
        graph.add_edge(x_proportion, multiply_x, to_port=0)
        graph.add_edge(input_shape, multiply_x, from_port="x_size", to_port=1)

        multiply_y = MultiplyOperator()
        graph.add_edge(y_proportion, multiply_y, to_port=0)
        graph.add_edge(input_shape, multiply_y, from_port="y_size", to_port=1)

        env_shape_constructor = ConstructorOperator()
        graph.add_edge(env_shape_type, env_shape_constructor, to_port=0)
        graph.add_edge(multiply_y, env_shape_constructor, to_port=1)
        graph.add_edge(multiply_x, env_shape_constructor, to_port=2)
        graph.add_edge(env_shape_constructor, output_node)

        return graph
