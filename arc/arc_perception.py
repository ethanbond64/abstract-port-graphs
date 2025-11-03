import numpy as np

from arc.arc_objects import ArcObject
from arc.arc_utils import zoom_to_non_zero_bounding_box_with_coordinates

CAVITY_COLOR = 21

def individual_cells(input_matrix, void_color=0):
    all_objects = []
    index = 0
    total_zones = input_matrix.shape[0] * input_matrix.shape[1]
    for i, row in enumerate(input_matrix):
        for j, col in enumerate(row):
            if col != void_color and col != 0:
                mask = np.zeros(input_matrix.shape, dtype=int)
                mask[i, j] = col
                zone_mask = np.array([[col]])
                all_objects.append(ArcObject(mask, individual_cells))

            index += 1

    return all_objects


def adjacency(input_matrix, void_color=0):
    return list(filter(lambda o: o.identity.color.value != void_color, find_adjacent_objects(input_matrix, adjacency)))


def adjacency_cardinal(input_matrix, void_color=0):
    return list(filter(lambda o: o.identity.color.value != void_color,
                       find_adjacent_objects(input_matrix, adjacency_cardinal, include_diagonals=False)))

def adjacency_any_color(input_matrix, void_color=0):
    # NOTE wrapping the find in this map which passes the mask back through the arcobject constructor because:
    # The find_adjacent_objects treats the "permission color" and its absence as a cavity, so we never actually zoom
    # past it. At the end of that method, the cavity color is replaced with 0, but this is after the zooming has already
    # Happened, leading to an invalid zoomed mask for our case.
    return list(map(lambda o: ArcObject(o.mask, perception_function=adjacency_any_color),
                    filter(lambda o: o.identity.color.value != void_color,
                           find_adjacent_objects(input_matrix, adjacency_any_color, permission_color=0))))

def adjacency_any_color_cardinal(input_matrix, void_color=0):
    # NOTE wrapping the find in this map which passes the mask back through the arcobject constructor because:
    # The find_adjacent_objects treats the "permission color" and its absence as a cavity, so we never actually zoom
    # past it. At the end of that method, the cavity color is replaced with 0, but this is after the zooming has already
    # Happened, leading to an invalid zoomed mask for our case.
    return list(map(lambda o: ArcObject(o.mask, perception_function=adjacency_any_color_cardinal),
                    filter(lambda o: o.identity.color.value != void_color,
                           find_adjacent_objects(input_matrix, adjacency_any_color, include_diagonals=False, permission_color=0))))

def all_cavities(input_matrix, void_color=0):
    return cavities_base(input_matrix, all_cavities)


def inner_cavities(input_matrix, void_color=0):
    return cavities_base(input_matrix, inner_cavities, include_inner=True, include_outer=False)

def outer_cavities(input_matrix, void_color=0):
    return cavities_base(input_matrix, outer_cavities, include_inner=False, include_outer=True)


def color_based_dividers_factory(color):
    return lambda mat, v: color_based_dividers(color, mat, v)


def color_based_dividers(divider_color, input_matrix, void_color=0):
    return_objects = []

    # Find objects with the divider color, add them to the set
    divider_objects = filter(lambda o: o.identity.color.value == divider_color,
                             find_adjacent_objects(input_matrix, color_based_dividers, include_diagonals=False))
    return_objects.extend(divider_objects)

    # Join all adjacent squares that are non-divider color and return them as objects
    non_divider_objects = list(filter(lambda o: o.identity.color.value != divider_color,
                                      find_adjacent_objects(input_matrix, color_based_dividers, include_diagonals=False,
                                                            permission_color=divider_color, void_color=void_color)))

    # Cut divided environments, so they are the entirety of their mask
    total_zones = len(non_divider_objects)
    for zone_index, zone_obj in enumerate(non_divider_objects):
        zone_obj.mask[zone_obj.mask == void_color] = 0  # TODO why is this necessary?
        zone_obj.identity.zoomed_mask[zone_obj.identity.zoomed_mask == void_color] = 0  # TODO fragile implementation
        # obj.rebuild_mask(obj.identity.zoomed_mask.shape)
        new_object = ArcObject.create_zone_object(zone_obj.mask, color_based_dividers,
                                                  zone_index=zone_index, total_zones=total_zones,
                                                  zone_mask=zone_obj.identity.zoomed_mask)
        return_objects.append(new_object)

    return return_objects


# ### Helper functions ###

def cavities_base(input_matrix, source_function, include_inner=True, include_outer=True, void_color=0):
    zoomed_mask, coordinates = zoom_to_non_zero_bounding_box_with_coordinates(input_matrix)
    zoomed_mask[zoomed_mask == 0] = CAVITY_COLOR

    confirmed_cavities = []
    potential_cavities = filter(lambda o: o.identity.color.value == CAVITY_COLOR,
                                find_adjacent_objects(zoomed_mask, None, include_diagonals=False))
    for cavity in potential_cavities:
        is_outer = any([cavity.location.x_location.min_boundary, cavity.location.x_location.max_boundary,
                        cavity.location.y_location.min_boundary, cavity.location.y_location.max_boundary])
        if include_inner and not is_outer:
            confirmed_cavities.append(cavity)
        elif include_outer and is_outer:
            confirmed_cavities.append(cavity)

    objects = []

    for cavity in confirmed_cavities:
        mask = np.zeros(input_matrix.shape, dtype=int)
        mask[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3]] = cavity.mask
        objects.append(ArcObject(mask, source_function))

    return objects


def find_adjacent_objects(matrix, source_function, include_diagonals=True, permission_color=None, void_color=None):
    bg_color = 0

    if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
        return []

    def dfs(dfs_i, dfs_j, value, dfs_mask):
        if dfs_i < 0 or dfs_i >= len(matrix) or dfs_j < 0 or dfs_j >= len(matrix[0]):
            return

        if permission_color is not None:
            color_matches = (matrix[dfs_i][dfs_j] == value and value == permission_color) or (
                    matrix[dfs_i][dfs_j] != permission_color and value != permission_color and value != bg_color)
        else:
            color_matches = (matrix[dfs_i][dfs_j] == value)

        if not color_matches or dfs_mask[dfs_i][dfs_j] != bg_color:
            return

        value = matrix[dfs_i][dfs_j]
        if permission_color is not None and value == bg_color and permission_color != bg_color:
            value = CAVITY_COLOR
        dfs_mask[dfs_i][dfs_j] = value

        # Define the directions for DFS
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        # Include diagonal directions if toggled on
        if include_diagonals:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals

        for di, dj in directions:
            dfs(dfs_i + di, dfs_j + dj, value, dfs_mask)

    def find_mask(search_i, search_j, value):
        new_mask = [[bg_color for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        dfs(search_i, search_j, value, new_mask)
        # if permission_color is not None:
        # new_mask = [[bg_color if x == CAVITY_COLOR else x for x in row] for row in new_mask]
        return new_mask

    objects: list[ArcObject] = []
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    length = len(matrix)
    width = len(matrix[0])
    for i in range(length):
        for j in range(width):
            if matrix[i][j] != bg_color and not visited[i][j]:
                mask = find_mask(i, j, matrix[i][j])

                # Mark all the cells in this object as visited
                for x in range(length):
                    for y in range(width):
                        if mask[x][y] != bg_color:
                            visited[x][y] = True

                obj = ArcObject(mask, source_function)
                if permission_color is not None:

                    # TODO I might be able to do all of this above...
                    # Check if this is a zone object (no permission color)
                    # If it is, zoom replace the cavity color with 0 and use the zoomed mask as the zone mask

                    # Otherwise just replace the cavity color with 0

                    obj.identity.zoomed_mask[obj.identity.zoomed_mask == CAVITY_COLOR] = 0
                    obj.mask[obj.mask == CAVITY_COLOR] = 0
                objects.append(obj)

    return objects
