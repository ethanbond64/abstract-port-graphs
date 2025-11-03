import math
from copy import copy
from enum import Enum, IntEnum
from typing import List, Optional, Tuple, Any, Type, Callable, Union, Dict

import numpy as np

from port_graphs import PortGraph, Edge
from programs import PerceptionModel, Program
from base_library import Library
from nodes import InputNode, OutputNode, Node
from base_types import PerceivedType, DslSet, DslConstructor, DslType
from arc.arc_utils import (zoom_to_non_zero_bounding_box, compute_color, can_compress,
                               compress_matrix, get_crow_distance, hash_shape_to_int, expand_matrix)
from interpreter.interpreter import Interpreter

ARC_LIBRARY = Library()


class EnvironmentShape(PerceivedType):

    ENV_PERCEPTION_KEY = "env_perception"

    def __init__(self, y_size: int, x_size: int):
        # TODO NORM throw exception if args are non-zero (4 dec) floats.
        self.y_size = int(y_size)
        self.x_size = int(x_size)

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return [DslConstructor(cls, [("y_size", int), ("x_size", int)])]

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [("y_size", int), ("x_size", int)]

    def get_perception_function(self):
        return EnvironmentShape.ENV_PERCEPTION_KEY

    def to_tuple(self) -> Tuple[int, int]:
        return self.y_size, self.x_size

    @classmethod
    def of(cls, mat: np.ndarray):
        return cls(mat.shape[0], mat.shape[1])

    def __hash__(self):
        return hash((self.y_size, self.x_size))

    def __eq__(self, other):
        if other is None:
            return False
        return self.y_size == other.y_size and self.x_size == other.x_size
ARC_LIBRARY.add_value_type(EnvironmentShape)

class ArcImage:

    def __init__(self, array: np.ndarray, maintain_color = False):
        if not isinstance(array, np.ndarray):
            raise TypeError("Array must contain only numeric values (int or float).")

        # Cast to integer array if possible
        if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, bool):
            self.array = array.astype(int)
        elif np.issubdtype(array.dtype, np.floating):
            rounded = np.rint(array)
            if not np.allclose(array, rounded, atol=1e-4): # TODO hard coded tolerance
                raise ValueError("Array contains floats that are not close enough to integers.")
            self.array = rounded.astype(int)
        else:
            raise ValueError("Array must be of type int or float.")

        self.array_hash = hash_shape_to_int(self.array, 0, maintain_color)  # TODO hard coded bg color - might be fine here tho

    def __eq__(self, other):
        if not isinstance(other, ArcImage):
            return False
        return self.array_hash == other.array_hash and np.array_equal(self.array, other.array)

    def __hash__(self):
        return self.array_hash
ARC_LIBRARY.add_value_type(ArcImage)

class Orientation(IntEnum):
    """Dihedral group D4 orientations (rotations + reflections)."""
    IDENTITY = 0       # 0 degrees
    ROT90 = 1          # rotate 90° clockwise
    ROT180 = 2         # rotate 180°
    ROT270 = 3         # rotate 270°
    FLIP = 4           # flip over y-axis
    FLIP_ROT90 = 5     # flip then rot90
    FLIP_ROT180 = 6    # flip then rot180
    FLIP_ROT270 = 7    # flip then rot270
ARC_LIBRARY.add_value_type(Orientation)


# --- Group operations (D4) ---

# Multiplication table: composition[a][b] = result of applying a then b
# Precomputed from the dihedral group structure
_COMPOSE = [
    #  ID,  R90, R180, R270, F,   FR90, FR180, FR270
    [0,   1,   2,    3,    4,   5,    6,     7],   # ID
    [1,   2,   3,    0,    5,   6,    7,     4],   # R90
    [2,   3,   0,    1,    6,   7,    4,     5],   # R180
    [3,   0,   1,    2,    7,   4,    5,     6],   # R270
    [4,   7,   6,    5,    0,   3,    2,     1],   # F
    [5,   4,   7,    6,    1,   0,    3,     2],   # FR90
    [6,   5,   4,    7,    2,   1,    0,     3],   # FR180
    [7,   6,   5,    4,    3,   2,    1,     0],   # FR270
]
# ACTUAL FROM WIKIPEDIA
# 0	1	2	3	4	5	6	7
# 1	2	3	0	5	6	7	4
# 2	3	0	1	6	7	4	5
# 3	0	1	2	7	4	5	6
# 4	7	6	5	0	3	2	1
# 5	4	7	6	1	0	3	2
# 6	5	4	7	2	1	0	3
# 7	6	5	4	3	2	1	0



# Inverses of each orientation
_INVERSE = [
    0,  # ID
    3,  # R90 inverse is R270
    2,  # R180
    1,  # R270 inverse is R90
    4,  # Flip inverse is itself
    5,  # Flip+R90
    6,  # Flip+R180
    7,  # Flip+R270
]


def compose_orientation(a: Orientation, b: Orientation) -> Orientation:
    """Return the composition a followed by b."""
    return Orientation(_COMPOSE[a][b])


def inverse_orientation(a: Orientation) -> Orientation:
    """Return the inverse of orientation a."""
    return Orientation(_INVERSE[a])


def difference_orientation(o1: Orientation, o2: Orientation) -> Orientation:
    """Return the orientation mapping o1 to o2 (i.e., o1⁻¹ * o2)."""
    return compose_orientation(inverse_orientation(o1), o2)


def apply_orientation(image: np.ndarray, orientation: Orientation) -> np.ndarray:
    if orientation == Orientation.IDENTITY:
        return image
    elif orientation == Orientation.ROT90:
        return np.rot90(image, -1)
    elif orientation == Orientation.ROT180:
        return np.rot90(image, 2)
    elif orientation == Orientation.ROT270:
        return np.rot90(image, 1)
    elif orientation == Orientation.FLIP:
        return np.fliplr(image)
    elif orientation == Orientation.FLIP_ROT90:
        return np.rot90(np.fliplr(image), 1)
    elif orientation == Orientation.FLIP_ROT180:
        return np.rot90(np.fliplr(image), 2)
    elif orientation == Orientation.FLIP_ROT270:
        return np.rot90(np.fliplr(image), -1)


def rotate_vector_90(x: 'OneDimensionalLocationRelationship', y: 'OneDimensionalLocationRelationship'):
    return y.other_dimension().get_inverse(), x.other_dimension()


def reflect_vector(x: 'OneDimensionalLocationRelationship', y: 'OneDimensionalLocationRelationship', up_down=False):
    return (x, y.get_inverse()) if up_down else (x.get_inverse(), y)


def apply_orientation_vector(x: 'OneDimensionalLocationRelationship', y: 'OneDimensionalLocationRelationship',
                             orientation: Orientation):
    if orientation == Orientation.IDENTITY:
        return x, y
    elif orientation == Orientation.ROT90:
        return rotate_vector_90(x, y)
    elif orientation == Orientation.ROT180:
        return rotate_vector_90(*rotate_vector_90(x, y))
    elif orientation == Orientation.ROT270:
        return rotate_vector_90(*rotate_vector_90(*rotate_vector_90(x, y)))
    elif orientation == Orientation.FLIP:
        return reflect_vector(x, y, up_down=True)
    elif orientation == Orientation.FLIP_ROT90:
        return rotate_vector_90(*reflect_vector(x, y))
    elif orientation == Orientation.FLIP_ROT180:
        return rotate_vector_90(*rotate_vector_90(*reflect_vector(x, y)))
    elif orientation == Orientation.FLIP_ROT270:
        return rotate_vector_90(*rotate_vector_90(*rotate_vector_90(*reflect_vector(x, y))))


class ColorPatternSchema(Enum):
    CARDINAL = 0
    SPHERICAL = 1
    RADIAL = 2


class ColorValue:

    def __init__(self, value=None, source_mask:Optional[ArcImage]=None):
        self.value = value
        self.solid = True if value is None or isinstance(value, bool) else math.isclose(value, round(value), abs_tol=0.0001) # TODO hard coded tolerance
        self.source_mask = source_mask
        # self.normalized_mask = None
        # self.orientation = None
        # if not self.solid or (source_mask is not None and np.unique(source_mask.array[source_mask.array != 0]).size > 1):
        #     # TODO generate color orientations for non-square objects
        #     if source_mask is not None and source_mask.array.shape[0] == source_mask.array.shape[1]:
        #         shape_to_use = source_mask.array
        #         # compressible, size = can_compress(shape_to_use, 0)
        #         # if compressible:
        #         #     shape_to_use = compress_matrix(shape_to_use, size, 0)
        #
        #         normalized_arc_image, orientation = normalize_zoomed_mask_2(shape_to_use, maintain_color=True)
        #         self.normalized_mask = normalized_arc_image
        #         self.orientation = orientation
        #         # self.scale = size # TODO NEED SCALE!
        #
        # # Cases:
        # 1. Integer prior value
        # 2. MultiColored texture
        # 3. Pattern patch
        # 4. Radial pattern


        self.pattern_schema = None # TODO make this work with orientation...
        self.radial_dict = None

    # @staticmethod
    # def constructor_norm_orientation(norm_mask: ArcImage, orientation: int):
    #     # TODO compose the value from the norm_mask and orientation
    #     actual_mask = apply_orientation(norm_mask.array, orientation)
    #     return ColorValue.from_zoomed_mask(actual_mask)
    #
    # @staticmethod
    # def applicable_constructor_norm_orientation(color: 'ColorValue') -> bool:
    #     if not color.solid and color.normalized_mask is not None and color.orientation is not None:
    #         # TODO restricted to rectangles for now!
    #         if (color.normalized_mask.array == 0).sum() == 0:
    #             return True
    #     return False

    @staticmethod
    def from_pattern_patch(pattern_patch):
        val = ColorValue.from_zoomed_mask(pattern_patch)
        val.pattern_schema = ColorPatternSchema.CARDINAL
        return val

    @staticmethod
    def from_radial_dict(radial_dict):
        val = ColorValue(1.5)#hash(tuple(sorted(radial_dict.values()))))
        val.pattern_schema = ColorPatternSchema.RADIAL
        val.radial_dict = radial_dict
        return val

    @staticmethod
    def from_zoomed_mask(zoomed_mask):
        color_value = compute_color(zoomed_mask, 0)
        # TODO if there are multiple colors, we need to store a number that represents color in locations,
        #  color not in locations, color by existence of color...
        return ColorValue(color_value, source_mask=ArcImage(zoomed_mask, maintain_color=True))

    def __eq__(self, __value):
        return self.value == __value.value

    def __hash__(self):
        return hash(self.value)
ARC_LIBRARY.add_value_type(ColorValue)

EMPTY_COLOR = ColorValue(0)


class ShapeValue(DslType):

    def __init__(self, normalized_shape=None, orientation = None, scale=None, source_mask=None):
        self.value = hash((normalized_shape, orientation, scale))
        self.normalized_shape = normalized_shape
        self.orientation = orientation
        self.scale = scale
        self.source_mask = source_mask

        if source_mask is None and normalized_shape is not None and orientation is not None and scale is not None:
            # Rotate the norm image
            rotated_matrix = apply_orientation(normalized_shape.array, orientation)
            # Scale it up
            scaled_matrix = expand_matrix(rotated_matrix, self.scale, 0) # TODO hard coded bg color
            self.source_mask = scaled_matrix

        self.bounding_width = None if source_mask is None else source_mask.shape[1]
        self.bounding_height = None if source_mask is None else source_mask.shape[0]
        self.bounding_area = None if self.bounding_width is None or self.bounding_height is None else (
                self.bounding_width * self.bounding_height)

        # IMPORTANT FROM PRIORS
        if self.bounding_area is not None:
            self.squareness = self.bounding_width / self.bounding_height \
                if self.bounding_height > self.bounding_width else self.bounding_height / self.bounding_width
        else:
            self.squareness = None

        self.symmetric_x = None if self.source_mask is None else ShapeValue.is_symmetric_x(self.source_mask)
        self.symmetric_y = None if self.source_mask is None else ShapeValue.is_symmetric_y(self.source_mask)
        self.count = None if self.source_mask is None else np.count_nonzero(self.source_mask)

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return [DslConstructor(cls, [
            ("normalized_shape", ArcImage),
            ("orientation", Orientation),
            ("scale", int)
        ])]

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [
            ("normalized_shape", ArcImage),
            ("orientation", Orientation), # NOTE: ONLY COMPARE IN SYNTHESIS IF NORM SHAPE IS THE SAME - OTHERWISE NOISE
            ("scale", int),
            ("bounding_width", int),
            ("bounding_height", int),
            ("bounding_area", int),
            ("count", int),
            ("symmetric_x", bool),
            ("symmetric_y", bool)
        ]

    @staticmethod
    def from_zoomed_mask(zoomed_mask, zone=False):
        size = 1
        shape_to_use = zoomed_mask
        if not zone:
            compressible, size = can_compress(zoomed_mask, 0)
            if compressible:
                shape_to_use = compress_matrix(zoomed_mask, size, 0)

        normalized_shape, orientation = normalize_zoomed_mask_2(shape_to_use)
        size = size if size is not None else 1
        return ShapeValue(normalized_shape, orientation, size, zoomed_mask)

    @staticmethod
    def is_symmetric_x(zoomed_mask) -> bool:
        colorless = zoomed_mask.copy()
        colorless[colorless != 0] = 1
        return np.all(colorless == np.flip(colorless, 0))

    @staticmethod
    def is_symmetric_y(zoomed_mask) -> bool:
        colorless = zoomed_mask.copy()
        colorless[colorless != 0] = 1
        return np.all(colorless == np.flip(colorless, 1))

    def __eq__(self, __value):
        if not isinstance(__value, ShapeValue):
            return False
        return self.value == __value.value  # TODO not robust?

    def __hash__(self):
        return self.value
ARC_LIBRARY.add_value_type(ShapeValue)


# TODO need to figure out what to do about maintain color...
def normalize_zoomed_mask_2(mask: np.ndarray, maintain_color = False) -> Tuple[ArcImage, Orientation]:

    # The mask starts as if it is IDENTITY
    all_oriented_masks = []
    for starter_orientation in Orientation:
        oriented_mask = apply_orientation(mask, starter_orientation)
        arc_image = ArcImage(oriented_mask, maintain_color)
        # arc_image_with_color = ArcImage(oriented_mask, True)
        all_oriented_masks.append((arc_image, arc_image.array_hash, starter_orientation))

    # Find the minimum hash version, this is the true identity mask
    min_arc_image, _, min_orientation = min(all_oriented_masks, key=lambda x: (x[0].array_hash, x[1], x[2]))

    # if maintain_color and  mask.shape == (4, 4):
    #     print("PAUSE")
    #     print(min_arc_image.array)

    # Now that the min image is the new identity, the orientation of the original mask is the inverse.
    return min_arc_image, inverse_orientation(min_orientation)


EMPTY_SHAPE = ShapeValue(ArcImage(np.array([[0]])),0,0, np.array([[0]]))


class Direction(Enum):
    DIRECTION_X = 0
    DIRECTION_Y = 1


class Side(Enum):
    SIDE_MIN = 0
    SIDE_MAX = 1


class LocationSchema(Enum):
    CLOSE_SIDES = 0
    CENTERS = 1

class LocationSpecificAttribute:
    CENTER = 0
    RELATIVE_MIN = 1
    RELATIVE_MAX = 2
    ACTUAL_MIN = 3
    ACTUAL_MAX = 4

class LinearLocation(DslType):

    def __init__(self, center=None, actual_min=None, actual_max=None, relative_min=None, relative_max=None,
                 min_boundary=None, max_boundary=None, linear_env=None, env_shape: EnvironmentShape=None):
        self.center = center
        self.actual_min = actual_min
        self.actual_max = actual_max
        self.relative_min = relative_min
        self.relative_max = relative_max
        self.min_boundary = min_boundary
        self.max_boundary = max_boundary
        self.boundary = 1 if min_boundary or max_boundary else 0
        self.linear_env = linear_env
        self.env_shape = env_shape

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return [
            DslConstructor(cls.constructor_center, [("center", float)]),
            DslConstructor(cls.constructor_actual_min, [("actual_min", int)]),
            DslConstructor(cls.constructor_actual_max, [("actual_max", int)]),
            DslConstructor(cls.constructor_relative_min, [("relative_min", float)]),
            DslConstructor(cls.constructor_relative_max, [("relative_max", float)])
        ]

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [
            ("center", float),
            ("actual_min", int),
            ("actual_max", int),
            ("relative_min", float),
            ("relative_max", float)
        ]

    @classmethod
    def constructor_center(cls, center: float) -> 'LinearLocation':
        return cls(center=center)

    @classmethod
    def constructor_actual_min(cls, actual_min: int) -> 'LinearLocation':
        return cls(actual_min=actual_min)

    @classmethod
    def constructor_actual_max(cls, actual_max: int) -> 'LinearLocation':
        return cls(actual_max=actual_max)

    @classmethod
    def constructor_relative_min(cls, relative_min: float) -> 'LinearLocation':
        return cls(relative_min=relative_min)

    @classmethod
    def constructor_relative_max(cls, relative_max: float) -> 'LinearLocation':
        return cls(relative_max=relative_max)

    def get_bounds(self, linear_size, env):

        if self.center is not None:
            x_center = self.center * env * 1.001
            x_half = linear_size / 2
            return max(int(x_center - x_half), 0), int(x_center + x_half)

        if self.actual_min is not None:
            return self.actual_min, self.actual_min + linear_size

        if self.actual_max is not None:
            # # TODO confirm this edge case handling!
            functional_actual_max = self.actual_max
            # if linear_size == 1:
            #     functional_actual_max = functional_actual_max + 1
            return max(functional_actual_max - linear_size, 0), functional_actual_max

        if self.relative_min is not None:
            actual_min = int(self.relative_min * env)
            return actual_min, actual_min + linear_size

        if self.relative_max is not None:
            actual_max = int(self.relative_max * env)
            return max(actual_max - linear_size, 0), actual_max

        raise Exception("Not enough info for bounds")

    def get_env(self) -> Optional[EnvironmentShape]:
        return copy(self.env_shape)

    def refresh_location(self, direction, linear_size, env_shape: EnvironmentShape):

        if self.env_shape is not None and self.env_shape.x_size is not None and self.env_shape.y_size is not None:
            new_env_proportion = round(env_shape.y_size / env_shape.x_size, 4)
            original_env_proportion = round(self.env_shape.y_size / self.env_shape.x_size, 4)
            if new_env_proportion != original_env_proportion:
                scale = new_env_proportion / original_env_proportion
                self.center = self.center * scale if self.center is not None else None
                self.relative_min = self.relative_min * scale if self.relative_min is not None else None
                self.relative_max = self.relative_max * scale if self.relative_max is not None else None

        env_value = env_shape.y_size if direction == Direction.DIRECTION_Y else env_shape.x_size

        if self.center is None and self.actual_min is not None and self.actual_max is not None:
            self.center = ((self.actual_min + self.actual_max) / 2) / env_value
        elif self.center is None and self.actual_min is not None:
            self.center = (self.actual_min + (linear_size / 2)) / env_value
        elif self.center is None and self.actual_max is not None:
            self.center = (self.actual_max - (linear_size / 2)) / env_value

        if self.center is not None and (self.actual_min is None or self.actual_max is None):
            self.actual_min, self.actual_max = self.get_bounds(linear_size, env_value)

    # TODO check more than just center... might need a dedicated equals with a schema passed in.
    def __eq__(self, __value):
        if self.center is not None and __value.center is not None:
            delta = self.center - __value.center
        elif self.actual_min is not None and __value.actual_min is not None:
            delta = self.actual_min - __value.actual_min
        elif self.actual_max is not None and __value.actual_max is not None:
            delta = self.actual_max - __value.actual_max
        elif self.relative_max is not None and __value.relative_max is not None:
            delta = self.relative_max - __value.relative_max
        elif self.actual_min is not None and __value.actual_min is not None:
            delta = self.actual_min - __value.actual_min
        else:
            delta = 100
        return abs(delta) < 0.00001

    def __hash__(self):
        return hash(self.center)
ARC_LIBRARY.add_value_type(LinearLocation)

class LocationValue(DslType):

    def __init__(self, x_location: LinearLocation = None, y_location: LinearLocation = None):
        self.x_location = x_location
        self.y_location = y_location

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return [DslConstructor(cls, [
            ("x_location", LinearLocation),
            ("y_location", LinearLocation)
        ])]

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [
            ("x_location", LinearLocation),
            ("y_location", LinearLocation)
        ]

    def get_placement_bounds(self, obj_y_size, obj_x_size, env_y, env_x):
        x_min, x_max = self.x_location.get_bounds(obj_x_size, env_x)
        y_min, y_max = self.y_location.get_bounds(obj_y_size, env_y)
        return x_min, x_max, y_min, y_max

    def get_dimension_location(self, dimension):
        if dimension == Direction.DIRECTION_X:
            return self.x_location
        elif dimension == Direction.DIRECTION_Y:
            return self.y_location
        else:
            raise Exception("Invalid dimension")

    def set_dimension_location(self, dimension: Direction, location: LinearLocation):
        if dimension == Direction.DIRECTION_X:
            self.x_location = location
        elif dimension == Direction.DIRECTION_Y:
            self.y_location = location
        else:
            raise Exception("Invalid dimension")

    # TODO definitely not robust - use the source env size to determine if relative or abs values should be used
    def __eq__(self, __value):

        if not isinstance(__value, LocationValue):
            return False

        return self.x_location == __value.x_location and self.y_location == __value.y_location

    def __hash__(self):
        return hash((self.x_location, self.y_location))
ARC_LIBRARY.add_value_type(LocationValue)


CENTERED = LocationValue(LinearLocation(center=0.5, min_boundary=1, max_boundary=1),
                         LinearLocation(center=0.5, min_boundary=1, max_boundary=1))


class ObjectIdentity(DslType):

    def __init__(self, color_value: ColorValue, shape_value: ShapeValue, zoomed_mask=None):
        self.zoomed_mask = zoomed_mask
        self.color = color_value
        self.shape = shape_value
        self.norm_image = None
        self.orientation = None
        if zoomed_mask is not None:
            self.refresh_norm_identity()

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return [DslConstructor(cls.from_components, [
            ("color", ColorValue),
            ("shape", ShapeValue)
        ])]

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [
            ("color", ColorValue),
            ("shape", ShapeValue)
        ]

    def rebuild_zoomed_mask(self):

        if self.shape.source_mask is None:
            raise Exception("Cannot build without shape source mask")

        self.zoomed_mask = np.zeros(self.shape.source_mask.shape, dtype=int)

        # NOTE - lazy loaded center variable for radial patterns
        center = None

        if self.color.value % 1 != 0:
            for row in range(self.zoomed_mask.shape[0]):
                for col in range(self.zoomed_mask.shape[1]):

                    src_row = row
                    src_col = col

                    if self.shape.source_mask[row, col] == 0:
                        continue

                    if self.color.pattern_schema == ColorPatternSchema.CARDINAL:
                        if src_row >= self.color.source_mask.array.shape[0]:
                            src_row = src_row % self.color.source_mask.array.shape[0]
                        if src_col >= self.color.source_mask.array.shape[1]:
                            src_col = src_col % self.color.source_mask.array.shape[1]

                    # For spherical patterns, the source mask is a list, need to specify schema
                    # before reading from source mask to avoid index errors
                    if self.color.pattern_schema != ColorPatternSchema.RADIAL:
                        val = self.color.source_mask.array[src_row, src_col]
                    else:
                        # TODO need more control over where to specify where the center is
                        #  Should be an attribute of color within an identity...

                        # lazy load center
                        if center is None:
                            center = tuple(d //2 for d in self.zoomed_mask.shape)

                        distance = get_crow_distance(center, src_row, src_col)

                        val = self.color.radial_dict[distance]

                    self.zoomed_mask[row, col] = val
        else:
            self.zoomed_mask[self.shape.source_mask != 0] = self.color.value

        self.refresh_norm_identity()

    def refresh_norm_identity(self):
        norm_image, orientation = normalize_zoomed_mask_2(self.zoomed_mask, maintain_color=True)
        self.norm_image = norm_image
        self.orientation = orientation

    @classmethod
    def from_zoomed_mask(cls, zoomed_mask, zone=False):
        if zoomed_mask is None:
            return cls(ColorValue(), ShapeValue())
        shape = ShapeValue.from_zoomed_mask(zoomed_mask, zone)
        color = ColorValue(compute_color(zoomed_mask), source_mask=ArcImage(zoomed_mask, maintain_color=True))
        return cls(color, shape, zoomed_mask=zoomed_mask)

    @classmethod
    def from_components(cls, color_value, shape_value):
        identity = cls(color_value, shape_value)
        identity.rebuild_zoomed_mask()
        return identity

    def __eq__(self, __value):
        return self.shape == __value.shape and self.color == __value.color

    def __hash__(self):
        return hash((self.shape.__hash__(), self.color.__hash__()))
ARC_LIBRARY.add_value_type(ObjectIdentity)


EMPTY_IDENTITY = ObjectIdentity(EMPTY_COLOR, EMPTY_SHAPE)


class ZoneInfo:

    def __init__(self, zone_index, total_zones):
        self.zone_index = zone_index
        self.total_zones = total_zones

    def __eq__(self, other):
        return (self.zone_index, self.total_zones) == (other.zone_index, other.total_zones)

    def __hash__(self):
        return hash((self.zone_index, self.total_zones))


class ArcObject(PerceivedType):
    __global_id = 0

    @staticmethod
    def get_next_id():
        ArcObject.__global_id += 1
        return ArcObject.__global_id

    def __init__(self, mask, perception_function=None, parent_id=None, zone_info=None, zone_mask=None, z_index=0):
        # # TODO confirm these zoom args are the desired behavior - does the caller need to filter out empty objects?
        # # That is what the individual cells function does...
        # zoom = zone_info is None
        # zoom_location = zoom or zone_info.total_zones != 1

        self.id = ArcObject.get_next_id()
        self.mask = np.array(mask) if mask is not None else None
        self.zone_mask = np.array(zone_mask) if zone_mask is not None else None

        # Location use cases -
        # 1. non zone: get_object_location
        # 2. zone with zoomed mask: custom get location
        # 3. zone with single mask: centered
        # Dimension values
        if zone_info is None:
            self.location = get_object_location(self.mask)
        elif zone_mask is not None:
            self.location = get_object_location(self.mask, zone_mask)
        else:
            self.location = CENTERED

        # Zoomed mask (identity mask) use cases -
        # 1. non zone: zoom to bounding box
        # 2. zone with zoomed mask: use zoomed mask
        # 3. zone with single mask: use single mask
        zoomed_mask = None
        if self.mask is not None:
            if zone_info is None:
                zoomed_mask = zoom_to_non_zero_bounding_box(self.mask)
            elif zone_mask is not None:
                zoomed_mask = zone_mask
            else:
                zoomed_mask = self.mask

        self.identity: ObjectIdentity = ObjectIdentity.from_zoomed_mask(zoomed_mask)
        self.z_index = z_index

        self.zone_info: Optional[ZoneInfo] = zone_info

        # Input object attributes only
        self.__perception_function = perception_function
        self.parent_id = parent_id

    @classmethod
    def get_constructors(cls) -> List[DslConstructor]:
        return [
            DslConstructor(cls.constructor_1, [
                ("identity", ObjectIdentity),
                ("location", LocationValue)
            ]),
            DslConstructor(cls.constructor_1, [
                ("identity", ObjectIdentity),
                ("location", LocationValue),
                ("z_index", int)
            ])
        ]

    @classmethod
    def get_components(cls) -> List[Tuple[str, Type]]:
        return [
            ("identity", ObjectIdentity),
            ("location", LocationValue)
        ]

    def get_perception_function(self):
        return self.__perception_function

    @staticmethod
    def constructor_1(identity: ObjectIdentity, location: LocationValue, z_index=0):
        new_object = ArcObject(None, z_index=z_index)
        new_object.identity = identity
        new_object.location = location
        x_source_env = new_object.location.x_location.get_env()
        y_source_env = new_object.location.y_location.get_env()
        if x_source_env is not None and x_source_env == y_source_env:
            new_object.rebuild_mask(x_source_env)
        elif x_source_env is not None and y_source_env is None:
            new_object.rebuild_mask(x_source_env)
        elif x_source_env is None and y_source_env is not None:
            new_object.rebuild_mask(y_source_env)
        return new_object

    # TODO make external utility function
    # Internal use only for going from structured data back to perceived data.
    def rebuild_mask(self, env_shape: EnvironmentShape):
        new_mask = np.zeros(env_shape.to_tuple(), dtype=int)
        self.identity.rebuild_zoomed_mask()
        self.mask = add_identity_at_location(self.identity, self.location, new_mask)

    # TODO make external utility function
    def refresh_location_from_components(self, env_shape: EnvironmentShape):
        self.location.x_location.refresh_location(Direction.DIRECTION_X, self.identity.shape.bounding_width, env_shape)
        self.location.y_location.refresh_location(Direction.DIRECTION_Y, self.identity.shape.bounding_height, env_shape)

    def __eq__(self, other):
        return self.location == other.location and self.identity == other.identity

    def __hash__(self):
        return hash((self.identity, self.location))

    # TODO remove
    @classmethod
    def create_zone_object(cls, mask, perception_function=None, zone_index=1, total_zones=1, zone_mask=None,
                           parent_id=None):

        if zone_mask is None and total_zones > 1:
            raise Exception("Zone mask required for zone object with siblings")
        elif zone_mask is not None and total_zones == 1:
            raise Exception("Zone mask not required for zone object with no siblings")

        return cls(mask, perception_function, parent_id, zone_info=ZoneInfo(zone_index, total_zones),
                   zone_mask=zone_mask)
ARC_LIBRARY.add_value_type(ArcObject)


class CompositeArcObject(ArcObject):
    def __init__(self, child_objects: DslSet[ArcObject]):
        first_mask_size = list(child_objects)[0].mask.shape
        if not all(first_mask_size == o.mask.shape for o in child_objects):
            raise Exception("Composite arc child objects must have same mask shape")
        mask = np.zeros(first_mask_size)
        # TODO order and mismatched envs unhandleded
        for obj in child_objects:
            mask[obj.mask != 0] = obj.mask[obj.mask != 0]
        ArcObject.__init__(self, mask)


class ArcZone:

    def __init__(self, shape_img: Any, location: LocationValue, zone_index: int):  # TODO img data structure
        self.shape_img = shape_img
        self.location = location
        self.zone_index = zone_index

    def __eq__(self, other):
        return (self.location.x_location.actual_min == other.location.x_location.actual_min and
                self.location.x_location.actual_max == other.location.x_location.actual_max and
                self.location.y_location.actual_min == other.location.y_location.actual_min and
                self.location.y_location.actual_max == other.location.y_location.actual_max and
                self.shape_img.shape == other.shape_img.shape)

    def __hash__(self):
        # TODO this is not robust
        location_tuple = (self.location.x_location.actual_min, self.location.x_location.actual_max,
                          self.location.y_location.actual_min, self.location.y_location.actual_max)
        return hash((location_tuple, self.shape_img.shape))


class TwoDimensionalLocationRelationship:

    def __init__(self, x_location_relationship, y_location_relationship):
        if x_location_relationship.location_schema != y_location_relationship.location_schema:
            raise Exception("x_location_relationship and y_location_relationship must have same schema")
        self.location_schema = x_location_relationship.location_schema
        self.x_location_relationship: OneDimensionalLocationRelationship = x_location_relationship
        self.y_location_relationship: OneDimensionalLocationRelationship = y_location_relationship

    def __eq__(self, __value):

        if __value is None or not isinstance(__value, TwoDimensionalLocationRelationship):
            return False

        return (self.location_schema == __value.location_schema and
                self.x_location_relationship == __value.x_location_relationship and
                self.y_location_relationship == __value.y_location_relationship)

    def __hash__(self):
        return hash((self.location_schema, self.x_location_relationship, self.y_location_relationship))


class OneDimensionalLocationRelationship:

    def __init__(self, location_schema: LocationSchema, dimension: Direction, side: Optional[Side], distance, aligned=False):
        self.location_schema = location_schema
        self.dimension = dimension
        self.side = side
        self.distance = round(distance, 3)
        self.aligned = aligned

    def other_dimension(self):
        other_dimension = Direction.DIRECTION_Y if self.dimension == Direction.DIRECTION_X else Direction.DIRECTION_X
        return OneDimensionalLocationRelationship(self.location_schema, other_dimension, self.side, self.distance, self.aligned)

    def get_inverse(self):
        other_side = None if self.side is None else Side.SIDE_MIN if self.side == Side.SIDE_MAX else Side.SIDE_MAX
        return OneDimensionalLocationRelationship(self.location_schema, self.dimension, other_side, -self.distance, self.aligned)

    @staticmethod
    def get_one_dimensional_relationship(location_1: LocationValue, location_2: LocationValue,
                                         dimension: Direction, location_schema: LocationSchema):

        linear_location_1 = location_1.get_dimension_location(dimension)
        linear_location_2 = location_2.get_dimension_location(dimension)

        if location_schema == LocationSchema.CLOSE_SIDES:
            return OneDimensionalLocationRelationship.get_close_sides_one_dim_rel(dimension, linear_location_1,
                                                                            linear_location_2)
        return OneDimensionalLocationRelationship.get_centers_one_dim_rel(dimension, linear_location_1, linear_location_2)


    @staticmethod
    def get_close_sides_one_dim_rel(dimension: Direction, linear_location_1: LinearLocation,
                                    linear_location_2: LinearLocation,) -> 'OneDimensionalLocationRelationship':

        if linear_location_1.center == linear_location_2.center:
            return OneDimensionalLocationRelationship(LocationSchema.CLOSE_SIDES, dimension, None, 0, True)

        min_delta = linear_location_1.actual_min - linear_location_2.actual_max
        max_delta = linear_location_1.actual_max - linear_location_2.actual_min

        aligned = False

        if min_delta == -max_delta:
            side = None
            distance = 0
            aligned = True
        elif min_delta > 0:
            side = Side.SIDE_MIN
            distance = -min_delta
        elif max_delta < 0:
            side = Side.SIDE_MAX
            distance = max_delta
        elif abs(min_delta) <= abs(max_delta):
            side = Side.SIDE_MIN
            distance = -min_delta
        else:
            side = Side.SIDE_MAX
            distance = max_delta

        return OneDimensionalLocationRelationship(LocationSchema.CLOSE_SIDES, dimension, side, distance, aligned)


    @staticmethod
    def get_centers_one_dim_rel(dimension: Direction, linear_location_1: LinearLocation, linear_location_2: LinearLocation,) -> 'OneDimensionalLocationRelationship':
        delta = linear_location_2.center - linear_location_1.center
        aligned = round(delta, 4) == 0

        # If the relationship rotates, the decimal will be invalid if the env is non-square
        env_size = linear_location_1.env_shape.x_size if dimension == Direction.DIRECTION_X else linear_location_1.env_shape.y_size
        normalized_delta = delta * env_size
        return OneDimensionalLocationRelationship(LocationSchema.CENTERS, dimension, None,
                                                  normalized_delta, aligned)

    def __eq__(self, __value):

        if __value is None or not isinstance(__value, OneDimensionalLocationRelationship):
            return False

        if self.location_schema != __value.location_schema or self.dimension != __value.dimension:
            return False

        if self.location_schema == LocationSchema.CENTERS:
            return self.distance == __value.distance
        else:
            return (self.distance == __value.distance and self.side == __value.side) or (self.aligned and __value.aligned)

    def __hash__(self):
        hashable_distance = 0 if self.aligned else self.distance
        hashable_side = None if self.aligned else self.side
        return hash((self.location_schema, self.dimension, hashable_distance, hashable_side))


class ConcreteIdentityRelationship:

    def __init__(self, object_identity_1: ObjectIdentity, object_identity_2: ObjectIdentity):
        if object_identity_1.shape.normalized_shape != object_identity_2.shape.normalized_shape:
            raise Exception("ArcObjects must have same normalized shape")
        # TODO uncomment when color and shape are handled properly together
        # if object_identity_1.color.value != object_identity_2.color.value:
        #     raise Exception("ArcObjects must have same color")

        # TODO validate color too somehow (non-int color is currently poorly
        #  handled with diff orientations of same identity)

        self.delta_orientation = difference_orientation(object_identity_1.shape.orientation, object_identity_2.shape.orientation)
        self.scale_factor = object_identity_2.shape.scale / object_identity_1.shape.scale


def get_object_location(mask, zoomed_mask=None) -> LocationValue:
    if mask is None:
        return LocationValue(LinearLocation(), LinearLocation())
    env_x = mask.shape[1]
    env_y = mask.shape[0]

    if zoomed_mask is None:
        non_zero_indices = np.argwhere(mask != 0)

        # Find bounds using min and max along each axis
        y_min, x_min = np.min(non_zero_indices, axis=0)
        y_max, x_max = np.max(non_zero_indices, axis=0)
    else:
        x_min, y_min, x_max, y_max = min_max_location_within_larger_matrix(mask, zoomed_mask)

    x_max += 1
    y_max += 1

    # Calculate relative bounds
    x_min_rel = x_min / env_x
    x_max_rel = x_max / env_x
    y_min_rel = y_min / env_y
    y_max_rel = y_max / env_y

    # Calculate center coordinates
    x_center = ((x_max + x_min) / 2) / env_x
    y_center = ((y_max + y_min) / 2) / env_y

    x_min_boundary = 1 if x_min == 0 else 0
    x_max_boundary = 1 if x_max == env_x else 0
    y_min_boundary = 1 if y_min == 0 else 0
    y_max_boundary = 1 if y_max == env_y else 0

    env_shape = EnvironmentShape(env_y, env_x)

    x_location = LinearLocation(x_center, x_min, x_max, x_min_rel, x_max_rel,
                                x_min_boundary, x_max_boundary, env_x, env_shape)
    y_location = LinearLocation(y_center, y_min, y_max, y_min_rel, y_max_rel,
                                y_min_boundary, y_max_boundary, env_y, env_shape)
    return LocationValue(x_location, y_location)


def get_empty_object():
    empty_object = ArcObject(None)
    empty_object.identity = EMPTY_IDENTITY
    empty_object.location = CENTERED
    return empty_object


EMPTY_OBJECT = get_empty_object()

def min_max_location_within_larger_matrix(large, small):

    large = np.asarray(large)
    small = np.asarray(small)

    l_rows, l_cols = large.shape
    s_rows, s_cols = small.shape

    if s_rows > l_rows or s_cols > l_cols:
        raise ValueError("small must be smaller than large")

    windows = np.lib.stride_tricks.sliding_window_view(large, (s_rows, s_cols))
    matches = np.all(windows == small, axis=(2, 3))

    # get the index of "true" within matches
    starting_indices = np.argwhere(matches)

    if len(starting_indices) > 1:
        raise ValueError("small must be unique within large")

    y_min, x_min = starting_indices[0]
    x_max = x_min + s_cols - 1
    y_max = y_min + s_rows - 1
    return int(x_min), int(y_min), int(x_max), int(y_max)


def add_identity_at_location(identity: ObjectIdentity, location: LocationValue, mask):
    x_min, x_max, y_min, y_max = location.get_placement_bounds(*identity.zoomed_mask.shape, *mask.shape)
    functional_zoomed_mask = identity.zoomed_mask

    if x_min == 0 and x_max < functional_zoomed_mask.shape[1]:
        new_start = functional_zoomed_mask.shape[1] - x_max
        functional_zoomed_mask = functional_zoomed_mask[:, new_start:]
    if y_min == 0 and y_max < functional_zoomed_mask.shape[0]:
        new_start = functional_zoomed_mask.shape[0] - y_max
        functional_zoomed_mask = functional_zoomed_mask[new_start:, :]
    if x_max > mask.shape[1]:
        functional_zoomed_mask = functional_zoomed_mask[:, :mask.shape[1] - x_max]
        x_max = mask.shape[1]
    if y_max > mask.shape[0]:
        functional_zoomed_mask = functional_zoomed_mask[:mask.shape[0] - y_max, :]
        y_max = mask.shape[0]

    mask[y_min:y_max, x_min:x_max][functional_zoomed_mask != 0] = functional_zoomed_mask[functional_zoomed_mask != 0]
    return mask


# NOTE THIS IS CURRENTLY ONLY USED FOR A SHAPE PATTERN. CAN IT SUPPORT MORE THAN THAT?
class ConcreteLinearPattern:

    def __init__(self, sub_objects: List[ArcObject], direction: Direction):
        self.sub_objects = sub_objects
        self.direction = direction

    def apply(self, count: int):
        raise NotImplementedError("ConcreteLinearPattern.apply") # TODO reconcile with the logic that uses this class

    def __eq__(self, __value):
        # TODO equals in order of arc object hashcode along the list, eq on direction
        return super().__eq__(__value)

    def __hash__(self):
        # Hashes of the arc objects sorted and in a tuple inorder, hash of dir in outer tuple.
        return super().__hash__()


class ArcEnvironment(PerceptionModel):

    def __init__(self, perception_functions: List[Callable], void_color=1):
        self.perception_functions = perception_functions
        self.void_color = void_color
        self.in_matrix_node = InputNode(EnvironmentShape, perception_qualifiers=EnvironmentShape.ENV_PERCEPTION_KEY)
        self.output_matrix_node = OutputNode(value_type=EnvironmentShape)

    # TODO remove this
    def input_matrix_node(self) -> Callable:
        return self.in_matrix_node

    def apply_perception(self, input_matrix) -> List[PerceivedType]:
        # "Perceived" the grid size, then the individual objects
        env_shape = EnvironmentShape(*input_matrix.shape)
        arc_objects = apply_object_def_functions_iterative(self.perception_functions, input_matrix, self.void_color)
        return arc_objects + [env_shape]


class ArcGraph(Program):

    def __init__(self, env: ArcEnvironment):
        super().__init__(env, PortGraph([env.output_matrix_node]))
        self.env = env

    # Wrapper to make the solvers file more readable
    def add_edge(self, from_node: Node, to_node: Node, from_port: str = None, to_port: int = 0) -> Edge:
        self.graph.add_edge(from_node, to_node, from_port, to_port)

    # Wrapper to make the solvers file more readable
    def add_generic_edge(self, from_node: Union[Node, List[Node]], to_node: Union[Node, List[Node]],
                         from_port: Union[str, List[str]] = None,
                         to_port: Union[int, list[int]] = 0):
        self.graph.add_generic_edge(from_node, to_node, from_port, to_port)

    def format_output_values(self, output_node_values: Dict[int, List[Any]]) -> Any:

        env_shape = output_node_values[self.env.output_matrix_node.id][0]
        output_matrix = np.zeros(env_shape.to_tuple())


        all_output_values = [v for vs in output_node_values.values() for v in vs]
        for output_object in sorted(all_output_values, key=lambda a: a.z_index if isinstance(a, ArcObject) else -1):
            if not output_object is EMPTY_OBJECT:
                if isinstance(output_object, ArcObject):
                    output_object.refresh_location_from_components(env_shape)
                    output_object.rebuild_mask(env_shape)
                    output_matrix[output_object.mask != 0] = output_object.mask[output_object.mask != 0]

        output_matrix[
            output_matrix == 0] = self.env.void_color  # TODO get this from graph state in case it is parameterized.

        return output_matrix


    # If the output grid shape node does not have any inbound edges, default to the input grid size via an edge.
    def lazy_create_default_env_shape_program(self):
        if len(self.graph.get_edges_to_by_id(self.env.output_matrix_node.id)) == 0:
            self.graph.add_edge(self.env.input_matrix_node(), self.env.output_matrix_node)


def full(input_matrix, void_color=0):
    new_matrix = np.zeros(input_matrix.shape)
    new_matrix[input_matrix != void_color] = input_matrix[input_matrix != void_color]
    return [ArcObject.create_zone_object(new_matrix, full)]


def apply_object_def_functions_iterative(object_def_functions, matrix, void_color):
    all_objects = []
    last_objects = []

    functional_void_color = 0 if void_color is None else void_color

    full_object = full(matrix, functional_void_color)
    all_objects.extend(full_object)
    last_objects.extend(full_object)

    # Remove full if it was manually passed in, because it is always automatically applied
    filtered_object_def_functions = [fn for fn in object_def_functions if fn != full]
    for fn in filtered_object_def_functions:

        new_objects = []

        grids = [matrix] if len(last_objects) == 0 else [o.identity.zoomed_mask for o in last_objects]
        for grid in grids:
            # functional_mask = obj.mask if fn not in ZONE_FUNCTIONS else obj.identity.zoomed_mask
            for new_object in fn(grid, functional_void_color):
                # a zone function
                new_object.parent_id = 0#obj.id
                new_objects.append(new_object)

        all_objects.extend(new_objects)

    return all_objects


class ArcInterpreter(Interpreter):

    def __init__(self, initial_depth=0):
        super().__init__(initial_depth)

    def validate_safe_recursion(self, recursive_value: Any) -> bool:

        if isinstance(recursive_value, ArcObject):
            env_x = recursive_value.location.x_location.linear_env
            env_y = recursive_value.location.y_location.linear_env
            recursive_value.refresh_location_from_components(EnvironmentShape(env_y, env_x))

            x_min, x_max, y_min, y_max = recursive_value.location.get_placement_bounds(
                recursive_value.identity.shape.bounding_height,
                recursive_value.identity.shape.bounding_width,
                env_y, env_x)

            if (x_min is None and x_max is None) or (y_min is None and y_max is None):
                raise Exception("Object existence validity unimplemented for non-actual locations")

            if (x_max is not None and x_max < 0) or (y_max is not None and y_max < 0):
                return False

            if (x_min is not None and x_min > env_x) or (y_min is not None and y_min > env_y):
                return False

        return True
