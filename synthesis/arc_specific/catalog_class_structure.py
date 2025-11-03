from typing import List, Dict, Type, Tuple, Callable, Any

from arc.arc_objects import ArcObject, ObjectIdentity, LocationValue, LinearLocation, ColorValue, \
    ShapeValue, EnvironmentShape, ArcZone, ArcImage, Orientation
from base_types import DslSet


class ClassConstructorMetadata:

    def __init__(self, constructor: Callable, components: List[Tuple[str, Type]],
                 reverse_applicable_function: Callable = lambda val: True):
        self.constructor = constructor
        self.components = components

        # Function which accepts a value, and returns a boolean determining if this
        # constructor is appropriate for the value.
        self.reverse_applicable_function = reverse_applicable_function

    def is_applicable(self, value: Any) -> bool:
        return self.reverse_applicable_function(value)


CLASS_CONSTRUCTOR_CATALOG: Dict[Type, List[ClassConstructorMetadata]] = {
    ArcObject: [
        ClassConstructorMetadata(ArcObject.constructor_1, [
            ("identity", ObjectIdentity),
            ("location", LocationValue),
        ])
                ],
    ObjectIdentity: [
        ClassConstructorMetadata(ObjectIdentity, [
            ("color", ColorValue),
            ("shape", ShapeValue)
        ])
    ],
    LocationValue: [
        ClassConstructorMetadata(LocationValue, [
            ("x_location", LinearLocation),
            ("y_location", LinearLocation)
        ]),
    ],
    ShapeValue: [
        ClassConstructorMetadata(ShapeValue, [
            ("normalized_shape", ArcImage),
            ("orientation", Orientation), # TODO restricted set of possible ints!
            ("scale", int)
        ])
    ],
    # ColorValue: [
    #     ClassConstructorMetadata(ColorValue, [
    #         ("normalized_mask", ArcImage),
    #         ("orientation", int),  # TODO restricted set of possible ints!
    #     ], ColorValue.applicable_constructor_norm_orientation),
    # ],
    LinearLocation: [
        ClassConstructorMetadata(LinearLocation.constructor_center, [
            ("center", float),
        ]),
        ClassConstructorMetadata(LinearLocation.constructor_actual_min, [
            ("actual_min", int),
        ]),
        ClassConstructorMetadata(LinearLocation.constructor_actual_max, [
            ("actual_max", int),
        ]),
        ClassConstructorMetadata(LinearLocation.constructor_relative_min, [
            ("relative_min", float),
        ]),
        ClassConstructorMetadata(LinearLocation.constructor_relative_max, [
            ("relative_max", float),
        ]),
    ],
    EnvironmentShape: [
        ClassConstructorMetadata(EnvironmentShape, [
            ("y_size", int),
            ("x_size", int)
        ]),
    ]
}


CLASS_COMPONENTS_CATALOG: Dict[Type, Dict[str, Type]] = {
    DslSet: {
        "size": int
    },
    EnvironmentShape: {
        "x_size": int,
        "y_size": int,
    },
    ArcZone: {
        "shape_img": Any,
        "location": LocationValue,
        "zone_index": int,
    },
    ArcObject: {
        "identity": ObjectIdentity,
        "location": LocationValue,
    },
    ObjectIdentity: {
        "color": ColorValue,
        "shape": ShapeValue
    },
    LocationValue: {
        "x_location": LinearLocation,
        "y_location": LinearLocation
    },
    ShapeValue: {
        "normalized_shape": ArcImage,
        "orientation": Orientation, # NOTE - DEPENDENT ON NORM SHAPE - CAN ONLY BE USED TO COMPARE IF NORM SHAPE IS THE SAME - OTHERWISE NOISE
        "scale": int,
        "bounding_width": int,
        "bounding_height": int,
        "bounding_area": int,
        "count": int,
        "symmetric_x": bool, # TODO need to support bools asap
        "symmetric_y": bool # TODO need to support bools asap
    },
    LinearLocation: {
        "center": float,
        "actual_min": int,
        "actual_max": int,
        "relative_min": float,
        "relative_max": float,
        # "min_boundary": int, TODO enable when you can track bayesian specificity... "less specific because only 2 possibl values"
        # "max_boundary": int # NOTE missing env shape attributes
    }
}

# Flattened tree traversal of ^.
# Key is root type (ArcObject, EnvironmentShape, etc)
# Value is list (in order of breadth first traversal) of key PATH (concatenated with ".") and type.
CLASS_COMPONENTS_CATALOG_FLATTENED: Dict[Type, List[Tuple[str, Type]]] = {}

ROOT_TYPES = [ArcObject, EnvironmentShape] # TODO include: , ArcZone, ArcSet]

for root_type in ROOT_TYPES:

    # Setup list in main dict, don't want to use a default dict here
    CLASS_COMPONENTS_CATALOG_FLATTENED[root_type] = []

    queue = [(root_type, None)]  # (current_type, current_path)
    while queue:
        current_type, current_path = queue.pop(0)

        CLASS_COMPONENTS_CATALOG_FLATTENED[root_type].append((current_path, current_type))

        next_level = CLASS_COMPONENTS_CATALOG.get(current_type, {})
        for component_path, component_type in next_level.items():
            path_base = "" if current_path is None else current_path + "."
            full_path = path_base + component_path
            queue.append((component_type, full_path))
