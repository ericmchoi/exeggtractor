""" utils.py - utility tools used in exeggtractor

This module contains some utility functions used in exeggtractor
"""
import math

import cv2
import numpy as np


def order_points(contour):
    """Order the points of a quadrilateral contour clockwise from the topleft"""
    if contour.shape[0] != 4:
        raise ValueError("contour does not have 4 points.")

    points = contour.reshape(4, 2)
    points = points[points[:, 1].argsort()]

    if points[0][0] > points[1][0]:
        points[[0, 1]] = points[[1, 0]]
    if points[2][0] < points[3][0]:
        points[[2, 3]] = points[[3, 2]]

    return points


def distance(a, b):
    """Calculate the euclidean distance between 2 points"""
    # pylint: disable=invalid-name
    return np.linalg.norm(a - b)


def crop_fixed_perspective(img, contour, dsize=None):
    """Crop a quadrilateral region of an image and straighten it"""
    points = order_points(contour)

    if dsize:
        width = dsize[0]
        height = dsize[1]
    else:
        width = max(distance(points[0], points[1]), distance(points[2], points[3]))
        height = max(distance(points[0], points[3]), distance(points[1], points[2]))

    new_points = [
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]

    dst = np.array(new_points, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(points.astype(np.float32), dst)
    return cv2.warpPerspective(img, mat, (int(width), int(height)))


def draw_hough_lines(image, lines, color, thickness=1):
    """draw lines returned from HoughLines onto an image"""
    # pylint: disable=invalid-name
    height, width = image.shape[:2]
    length = height + width

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
        pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))

        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)


def rectangularContour(x1, x2, y1, y2):
    """return rectangular contour defined by 2 horizontal and vertical coordinates"""
    # pylint: disable=invalid-name
    return np.array(
        [
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]],
        ],
        dtype=np.int32,
    )
