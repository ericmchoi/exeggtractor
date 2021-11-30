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
    return np.linalg.norm(a - b)


def crop_fixed_perspective(img, contour, dsize=None):
    """Crop a quadrilateral region of an image and straightens it"""
    points = order_points(contour)

    if dsize:
        width = dsize[0]
        height = dsize[1]
    else:
        width = max(distance(points[0], points[1]),
                    distance(points[2], points[3]))
        height = max(distance(points[0], points[3]),
                     distance(points[1], points[2]))

    new_points = [
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ]

    dst = np.array(new_points, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(points.astype(np.float32), dst)
    return cv2.warpPerspective(img, mat, (int(width), int(height)))
