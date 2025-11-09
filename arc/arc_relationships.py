import numpy as np

from arc.arc_objects import ArcObject
from nodes import RelationshipNode


class AlignedCardinal(RelationshipNode):

    def __init__(self):
        super().__init__([ArcObject, ArcObject])

    def apply(self, object_1: ArcObject, object_2: ArcObject):

        x_center_1 = object_1.location.x_location.center
        y_center_1 = object_1.location.y_location.center
        x_center_2 = object_2.location.x_location.center
        y_center_2 = object_2.location.y_location.center

        if x_center_1 == x_center_2 == y_center_1 == y_center_2 is None:
            raise Exception("AlignedCardinal requires location centers")

        return (x_center_1 == x_center_2) or (y_center_1 == y_center_2)

    def is_symmetrical(self) -> bool:
        return True

# Arg 1 is the object between the other two objects
class BetweenCardinal(RelationshipNode):

    def __init__(self):
        super().__init__([ArcObject, ArcObject, ArcObject])

    def apply(self, object_1: ArcObject, object_2: ArcObject, object_3: ArcObject):
        x_center_1 = object_1.location.x_location.center
        y_center_1 = object_1.location.y_location.center
        x_center_2 = object_2.location.x_location.center
        y_center_2 = object_2.location.y_location.center
        x_center_3 = object_3.location.x_location.center
        y_center_3 = object_3.location.y_location.center

        if x_center_1 == x_center_2 == y_center_1 == y_center_2 == x_center_3 == y_center_3 is None:
            raise Exception("BetweenCardinal requires location centers")

        if x_center_2 > x_center_1 > x_center_3:
            return True

        if x_center_2 < x_center_1 < x_center_3:
            return True

        if y_center_2 > y_center_1 > y_center_3:
            return True

        if y_center_2 < y_center_1 < y_center_3:
            return True

        return False


class Touches(RelationshipNode):

    def __init__(self):
        super().__init__([ArcObject, ArcObject])

    def apply(self, object_1: ArcObject, object_2: ArcObject):

        mask_a = (object_1.mask.copy() != 0)
        mask_b = (object_2.mask.copy() != 0)

        # If masks have a different size then this relationship is invalid
        if mask_a.shape != mask_b.shape:
            return False

        # If one of the objects is empty, they can't be touching.
        if not mask_a.any() or not mask_b.any():
            return False

        # Check for any neighbor overlap between the two objects.
        # We shift maskB by all offsets in {-1,0,1} (in both dimensions) and compare with maskA.
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # Determine slices for the rows based on dx
                if dx == -1:
                    A_rows = slice(1, None)  # Skip first row in A
                    B_rows = slice(0, -1)  # Skip last row in B
                elif dx == 0:
                    A_rows = slice(None)
                    B_rows = slice(None)
                else:  # dx == 1
                    A_rows = slice(0, -1)  # Skip last row in A
                    B_rows = slice(1, None)  # Skip first row in B

                # Determine slices for the columns based on dy
                if dy == -1:
                    A_cols = slice(1, None)  # Skip first column in A
                    B_cols = slice(0, -1)  # Skip last column in B
                elif dy == 0:
                    A_cols = slice(None)
                    B_cols = slice(None)
                else:  # dy == 1
                    A_cols = slice(0, -1)  # Skip last column in A
                    B_cols = slice(1, None)  # Skip first column in B

                # Check if there's any overlap between maskA and the shifted maskB.
                if np.any(mask_a[A_rows, A_cols] & mask_b[B_rows, B_cols]):
                    return True
        return False

    def is_symmetrical(self) -> bool:
        return True


class NotTouches(Touches):
    def __init__(self):
        super().__init__()

    def apply(self, object_1: ArcObject, object_2: ArcObject):
        return not super().apply(object_1, object_2)

    def is_symmetrical(self) -> bool:
        return True


# Returns true if the second arg object is completely within the bounding box of the first arg object
class ObjectContains(RelationshipNode):

    def __init__(self):
        super().__init__([ArcObject, ArcObject])

    def apply(self, larger_object: ArcObject, smaller_object: ArcObject):
        large_x_min = larger_object.location.x_location.actual_min
        large_x_max = larger_object.location.x_location.actual_max
        large_y_min = larger_object.location.y_location.actual_min
        large_y_max = larger_object.location.y_location.actual_max
        smaller_x_min = smaller_object.location.x_location.actual_min
        smaller_x_max = smaller_object.location.x_location.actual_max
        smaller_y_min = smaller_object.location.y_location.actual_min
        smaller_y_max = smaller_object.location.y_location.actual_max

        return ((large_x_min <= smaller_x_min) and (large_x_max >= smaller_x_max) and
                (large_y_min <= smaller_y_min) and (large_y_max >= smaller_y_max))
