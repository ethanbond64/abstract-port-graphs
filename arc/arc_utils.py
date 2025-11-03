import json
import os

import numpy as np


class ArcTask:

    def __init__(self, challenge_id, train, test):
        self.challenge_id = challenge_id
        self.train = [Case.of(case) for case in train]
        self.test = [Case.of(case) for case in test]

    @staticmethod
    def of(challenge_id, json_object):
        return ArcTask(challenge_id, json_object["train"], json_object["test"])

    @staticmethod
    def get(challenge_id):
        challenges = read_challenges("data/arc-agi_training_challenges.json")
        return challenges.get(challenge_id)


class Case:

    def __init__(self, input_matrix, output_matrix):
        self.input_matrix = np.array(input_matrix)
        self.output_matrix = np.array(output_matrix) if output_matrix is not None else None  # Nullable

    def set_output_matrix(self, output_matrix):
        self.output_matrix = np.array(output_matrix)

    @staticmethod
    def of(json_object):
        return Case(json_object["input"], json_object.get("output"))


def read_challenges(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_name) if not os.path.isabs(file_name) else file_name
    with open(file_path) as file:
        return {key: ArcTask.of(key, value) for key, value in json.load(file).items()}

def read_challenge_text(text):
    return {key: ArcTask.of(key, value) for key, value in json.loads(text).items()}

def read_solutions(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_name) if not os.path.isabs(file_name) else file_name
    with open(file_path) as file:
        return {key: value for key, value in json.load(file).items()}


def zoom_to_non_zero_bounding_box(matrix):
    return zoom_to_non_zero_bounding_box_with_coordinates(matrix)[0]


def zoom_to_non_zero_bounding_box_with_coordinates(matrix):

    # Get the indices of non-zero elements
    matrix = np.array(matrix)
    non_zero_indices = np.argwhere(matrix > 0)

    if non_zero_indices.size == 0:
        raise ValueError("The matrix contains no non-zero elements.")

    # Find the minimal and maximal row/column indices
    row_min, col_min = non_zero_indices.min(axis=0)
    row_max, col_max = non_zero_indices.max(axis=0)

    row_max += 1
    col_max += 1

    # Slice the matrix to the bounding box
    zoomed_matrix = matrix[row_min:row_max, col_min:col_max]

    return zoomed_matrix, (row_min, row_max, col_min, col_max)


def hash_shape_to_int(shape, bg_color, maintain_color=False):
    int_bg = int(bg_color)
    shape_tuple = tuple(
        tuple([int_bg if col == bg_color else (int(col) if maintain_color else 1)
               for col in row]) for row in shape)  # TODO hard coded 0
    shape_hash = hash(shape_tuple)
    # if maintain_color and shape.shape == (4, 4):
    #     print("AHHH")
    #     print(shape_tuple)
    #     print(shape_hash)
    return shape_hash


def rotate_90(shape):
    return np.rot90(shape, -1)


def rotate_90_reverse(shape):
    return np.rot90(shape, 1)


# Center is a tuple of (center row index, center col index)
def get_crow_distance(center, row, column, decimals=3):
    return get_crow_distance_explicit(row, column, *center, decimals)

def get_crow_distance_explicit(x1, y1, x2, y2, decimals=3):
    return round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), decimals)

def can_compress(matrix, bg_color):
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    if rows == 1 or cols == 1:
        return False, None

    gcd = np.gcd(rows, cols)
    common_denominators = sorted([i for i in range(1, gcd + 1) if gcd % i == 0 and i > 1], reverse=True)

    if len(common_denominators) == 0:
        return False, None

    # Find the smallest possible factor for compression
    for factor in common_denominators:
        block_rows = rows // factor
        block_cols = cols // factor
        for i in range(block_rows):
            for j in range(block_cols):
                sub_matrix = matrix[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
                if not (np.all(sub_matrix != bg_color) or np.all(sub_matrix == bg_color)):
                    break
            else:
                continue
            break
        else:
            return True, factor
    return False, None


def compress_matrix(matrix, factor, bg_color):
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    block_rows = rows // factor
    block_cols = cols // factor

    compressed_matrix = np.zeros((block_rows, block_cols), dtype=int)

    for i in range(block_rows):
        for j in range(block_cols):
            sub_matrix = matrix[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
            if np.all(sub_matrix != bg_color) and np.all(sub_matrix == sub_matrix[0, 0]):
                compressed_matrix[i, j] = sub_matrix[0, 0]
    return compressed_matrix


def expand_matrix(matrix, factor, bg_color):
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    expanded_matrix = np.full((rows * factor, cols * factor), bg_color, dtype=int)

    for i in range(rows):
        for j in range(cols):
            expanded_matrix[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = matrix[i, j]

    return expanded_matrix

def compute_color(matrix, bg_color=0):
    unique_colors = np.unique(matrix[matrix != bg_color])  # Get all unique non-empty colors

    if len(unique_colors) == 0:
        return bg_color

    # Check if all colors are the same
    if len(unique_colors) == 1:
        return unique_colors[0]

    # If not all colors are the same, proceed with computation
    rows, cols = matrix.shape

    # Create coordinate grids for i and j
    i_indices, j_indices = np.indices((rows, cols))

    # Compute distances from the center for weighting
    distances = np.sqrt(i_indices ** 2 + (j_indices + 0.001) ** 2)  # NOTE the + 0.001 is
    # to differentiate distance across dimensions
    weights = 1 / (1 + distances)

    # Calculate weighted sum of colors and total weight for non-zero elements
    color_weights = matrix * weights
    total_color_value = np.sum(color_weights[matrix != bg_color])
    total_weight = np.sum(weights[matrix != bg_color])

    # Calculate the weighted average color
    weighted_average_color = total_color_value / total_weight

    return weighted_average_color

