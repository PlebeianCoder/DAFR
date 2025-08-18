from __future__ import annotations

from typing import Any
from typing import TypeVar

import numpy as np

# This file can be considered as the Python equivalent of the MediaPipe procrustes_solver.cc file.
# Hence, we copied the comments that explain the algorithm.


# The type of the number of points
N = TypeVar("N", int, np.int64)


# The weighted problem is thoroughly addressed in Section 2.4 of:
# D. Akca, Generalized Procrustes analysis and its applications
# in photogrammetry, 2003, https://doi.org/10.3929/ethz-a-004656648

# Notable differences in the code presented here are:
# - In the paper, the weights matrix W_p is Cholesky-decomposed as Q^T Q.
#   Our W_p is diagonal (equal to diag(sqrt_weights^2)),
#   so we can just set Q = diag(sqrt_weights) instead.
# - In the paper, the problem is presented as
#   (for W_k = I and W_p = tranposed(Q) Q):
#   || Q (c A T + j tranposed(t) - B) || -> min.
#   We reformulate it as an equivalent minimization of the transpose's
#   norm:
#   || (c tranposed(T) tranposed(A) - tranposed(B)) tranposed(Q) || -> min,
#   where tranposed(A) and tranposed(B) are the source and the target point
#   clouds, respectively, c tranposed(T) is the rotation+scaling R sought
#   for, and Q is diag(sqrt_weights).
#
#   Most of the derivations are therefore transposed.


def solve_weighted_orthogonal_problem(
    source_points: np.ndarray[(3, N), np.dtype[np.float64]],
    target_points: np.ndarray[(3, N), np.dtype[np.float64]],
    weights: np.ndarray[(N, 1), np.dtype[np.float64]],
) -> np.ndarray[(4, 4), np.dtype[np.float64]]:
    """
    Solve the weighted orthogonal problem.
    This computes the optimal rotation, scale and translation to transform the source points to the target points.

    :param source_points: The source points.
    :param target_points: The target points.
    :param weights: The weights of each point pair.
    :return: The optimal transformation matrix.
    """
    assert source_points.shape == target_points.shape

    sqrt_weights = np.sqrt(weights, dtype=np.float64)

    # transposed(A_w)
    weighted_sources = source_points * sqrt_weights.T
    # transposed(B_w)
    weighted_targets = target_points * sqrt_weights.T

    # w = transposed(j_w) j_w
    total_weight = (sqrt_weights * sqrt_weights).sum()

    # Let C = (j_w transposed(j_w)) / (transposed(j_w) j_w).
    # Note that C = transposed(C), hence (I - C) = transposed(I - C).

    # transposed(A_w) C = transposed(A_w) j_w transposed(j_w) / w =
    # (transposed(A_w) j_w) transposed(j_w) / w = c_w transposed(j_w),
    # where c_w = transposed(A_w) j_w / w is a k x 1 vector calculated here:.
    twice_weighted_sources = weighted_sources * sqrt_weights.T
    source_center_of_mass = (
        np.sum(twice_weighted_sources, axis=1) / total_weight
    ).reshape(-1, 1)

    # transposed((I - C) A_w) = transposed(A_w) (I - C) =
    # transposed(A_w) - transposed(A_w) C = transposed(A_w) - c_w transposed(j_w)
    centered_weighted_sources = (
        weighted_sources - source_center_of_mass @ sqrt_weights.T
    )

    rotation = compute_optimal_rotation(weighted_targets @ centered_weighted_sources.T)

    scale = compute_optimal_scale(
        centered_weighted_sources, weighted_sources, weighted_targets, rotation
    )

    # R = c transposed(T)
    rotation_and_scale = scale * rotation

    # Compute optimal translation for the weighted problem.

    # transposed(B_w - C A_w T) = transposed(B_w) - R transposed(A_w) in (54)
    pointwise_diffs = weighted_targets - rotation_and_scale @ weighted_sources
    # Multiplication by j_w is a respectively weighted column sum. (54) from the paper
    weighted_pointwise_diffs = pointwise_diffs * sqrt_weights.T
    translation = weighted_pointwise_diffs.sum(axis=1) / total_weight

    return combine_transform_matrix(rotation_and_scale, translation)


def combine_transform_matrix(
    rotation_and_scale: np.ndarray[(3, 3), np.dtype[np.float64]],
    translation: np.ndarray[(3,), np.dtype[np.float64]],
) -> np.ndarray[(4, 4), np.dtype[np.float64]]:
    """
    Combine the rotation and scale matrix with the translation vector.

    :param rotation_and_scale: The rotation and scale matrix.
    :param translation: The translation vector.
    :return: The combined transformation matrix.
    """
    result = np.identity(4, dtype=np.float64)
    result[:3, :3] = rotation_and_scale
    result[:3, 3] = translation

    return result


# `design_matrix` is a transposed LHS of (51) in the paper.
def compute_optimal_rotation(
    design_matrix: np.ndarray[(3, 3), np.dtype[np.float64]],
) -> np.ndarray[(3, 3), np.dtype[np.float64]]:
    """
    Compute the optimal rotation matrix.

    :param design_matrix: The design matrix.
    :return: The optimal rotation matrix.
    """
    u, _, v = np.linalg.svd(design_matrix, full_matrices=True)
    post_rotation = u
    pre_rotation = v

    # Disallow reflection by ensuring that det(`rotation`) = +1 (and not -1).
    # see "4.6 Constrained orthogonal Procrustes problem" in the Gower & Dijksterhuis's book "Procrustes Analysis".
    # We flip the sign of the least singular value along with a column in W.

    # Note that now the sum of singular values doesn't work for scale estimation due to this sign flip
    if np.linalg.det(post_rotation) * np.linalg.det(pre_rotation) < 0:
        post_rotation[:, 2] *= -1

    # Transposed (52) from the paper.
    return post_rotation @ pre_rotation


def compute_optimal_scale(
    centered_weighted_sources: np.ndarray[(3, N), np.dtype[np.float64]],
    weighted_sources: np.ndarray[(3, N), np.dtype[np.float64]],
    weighted_targets: np.ndarray[(3, N), np.dtype[np.float64]],
    rotation: np.ndarray[(3, 3), np.dtype[np.float64]],
) -> float:
    """
    Compute the optimal scale.

    :param centered_weighted_sources: The centered weighted source points.
    :param weighted_sources: The weighted source points.
    :param weighted_targets: The weighted target points.
    :param rotation: The optimal rotation matrix.
    :return: The optimal scale.
    """
    # transposed(T) transposed(A_w) (I - C).
    rotated_centered_weighted_sources = rotation @ centered_weighted_sources

    # Use the identity trace(A B) = sum(A * B^T) to avoid building large intermediate matrices
    # (* is the Hadamard product, i.e. element-wise multiplication).
    # (53) from the paper.
    numerator = (rotated_centered_weighted_sources * weighted_targets).sum()
    denominator = (centered_weighted_sources * weighted_sources).sum()

    return numerator / denominator
