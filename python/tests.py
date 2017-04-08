from __future__ import print_function
import numpy as np


def angle_between_vectors_w_norms(v1, v2, normv1, normv2):
    theta = np.arccos(np.dot(v1, v2) / (normv1 * normv2))
    if np.isnan(theta):
        return 0
    return theta


def angular_upper_bound(item_norm, theta_ic, theta_uc_b):
    if (theta_ic - theta_uc_b < 0):
        return item_norm
    else:
        return item_norm * np.cos(theta_ic - theta_uc_b)


# checks if:
#       1) cos(theta_ui) < cos(theta_ic - theta_uc)
#       2) cos(theta_ui) < cos(theta_ic - theta_b)
def check_angular_bounds(user_id, user_weight, item_weights, theta_uc, theta_b,
                         theta_ics, theta_uc_errors, theta_b_errors):
    user_norm = np.linalg.norm(user_weight)
    for item_id, item_weight in enumerate(item_weights):
        item_norm = np.linalg.norm(item_weight)
        theta_ui = angle_between_vectors_w_norms(user_weight, item_weight,
                                                 user_norm, item_norm)
        theta_ic = theta_ics[item_id]
        if np.cos(theta_ui) > np.cos(theta_ic - theta_uc):
            print(
                '%d,%d,%f,%f,%f,%d'
                % (user_id, item_id, theta_ui, theta_ic, theta_uc, theta_ic - theta_uc < 0), file=theta_uc_errors)
        if np.cos(theta_ui) > np.cos(theta_ic - theta_b):
            print(
                '%d,%d,%f,%f,%f,%d'
                % (user_id, item_id, theta_ui, theta_ic, theta_b, theta_ic - theta_b < 0), file=theta_b_errors)


# Reverse role of theta_ic and theta_uc. Swapping them shouldn't
# make any difference in the upper bound calculated
def flipped_angular_upper_bound(item_norm, theta_ic, theta_uc_b):
    if (theta_uc_b - theta_ic < 0):
        return item_norm
    else:
        return item_norm * np.cos(theta_uc_b - theta_ic)


# Go through all items for a single user and its assigned centroid, check
# and make sure that the upper bound and the flipped upper bound are both
# greater than the actual dot product
def compute_and_check_upper_bounds(user_weight, item_weights, centroid):
    user_norm = np.linalg.norm(user_weight)
    centroid_norm = np.linalg.norm(centroid)
    theta_uc = np.arccos(
        np.dot(user_weight, centroid) / (user_norm * centroid_norm))
    for item_id, item_weight in enumerate(item_weights):
        item_norm = np.linalg.norm(item_weight)
        theta_ic = np.arccos(
            np.dot(item_weight, centroid) / (item_norm * centroid_norm))
        upper_bound = user_norm * angular_upper_bound(item_norm, theta_ic,
                                                      theta_uc)
        actual = np.dot(user_weight, item_weight)
        if actual > upper_bound:
            print('item_id = %d, upper_bound = %f, actual = %f' %
                  (item_id, upper_bound, actual))
        bad_upper_bound = user_norm * flipped_angular_upper_bound(
            item_norm, theta_ic, theta_uc)
        if actual > bad_upper_bound:
            print('item_id = %d, bad_upper_bound = %f, actual = %f' %
                  (item_id, bad_upper_bound, actual))


def check_bounds_for_missing_items(user_weight, item_weights, centroid,
                                   top_K_items, true_top_K_items,
                                   min_rating_top_K):
    true_top_K_set = set(true_top_K_items)
    top_K_set = set(top_K_items)
    missing_items = true_top_K_set - top_K_set
    user_norm = np.linalg.norm(user_weight)
    centroid_norm = np.linalg.norm(centroid)
    theta_uc = np.arccos(
        np.dot(user_weight, centroid) / (user_norm * centroid_norm))
    for item_id in missing_items:
        item_weight = item_weights[item_id]
        item_norm = np.linalg.norm(item_weight)
        theta_ic = np.arccos(
            np.dot(item_weight, centroid) / (item_norm * centroid_norm))
        upper_bound = user_norm * angular_upper_bound(item_norm, theta_ic,
                                                      theta_uc)
        actual = np.dot(user_weight, item_weight)
