import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from colour import delta_E
from pytesseract import image_to_string

from .utils import crop_fixed_perspective, order_points

TESSERACT_CONFIG = r"--psm 7 -c page_separator=''"
CORNER_TESTS = [
    {
        "x": 0,
        "y": 0,
        "expected_color": [
            86.58447,
            -45.359375,
            0.84375
        ],
        "threshold":10
    },
    {
        "x": 0,
        "y": -1,
        "expected_color": [
            71.25244,
            -22.265625,
            -34.046875
        ],
        "threshold":10
    },
    {
        "x": -1,
        "y": 0,
        "expected_color": [
            0,
            0,
            0
        ],
        "threshold":2
    },
    {
        "x": -1,
        "y": -1,
        "expected_color": [
            0,
            0,
            0
        ],
        "threshold":2
    }
]

Image = np.ndarray

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ImageType(str, Enum):
    SCREENSHOT = 'screenshot'
    PHOTO = 'photo'


@dataclass
class DebugImage:
    name: str
    image: Image


@dataclass
class Result:
    debug_images: list[DebugImage]
    team: dict = None
    error: str = None


def _get_image_type(image: Image):
    if image.shape[0:2] not in [(720, 1280), (1080, 1920)]:
        return ImageType.PHOTO

    for test in CORNER_TESTS:
        bgr_color = np.array([[image[test["x"], test["y"]]]])
        bgr_color = bgr_color.astype(np.float32) / 255

        lab_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2Lab)

        color_diff = delta_E(lab_color, np.array([[test["expected_color"]]]))

        if color_diff > test["threshold"]:
            print(test, lab_color)
            return ImageType.PHOTO

    return ImageType.SCREENSHOT


def _get_screen_contour(image):
    ratio = max(720/image.shape[0], 1280/image.shape[1])
    dsize = (int(ratio*image.shape[1]), int(ratio*image.shape[0]))

    resize = cv2.resize(image, dsize)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, 80, 240)

    contours, _ = cv2.findContours(canny, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    screen = np.zeros([4, 1, 2], dtype=np.int32)
    max_area = 0.5 * resize.shape[0] * resize.shape[1]
    for cnt in contours:
        if cv2.contourArea(cnt) > max_area:
            eps = 0.04 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, eps, True)
            if len(poly) == 4:
                screen = poly
                max_area = cv2.contourArea(poly)

    return (screen / ratio).astype(np.int32)


def _find_white_regions(image):
    total_area = image.shape[0] * image.shape[1]
    white = np.min(image, axis=2)
    blur = cv2.GaussianBlur(white, (5, 5), 1)
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])

    threshold = 255
    total_pixels = 0

    while threshold and total_pixels < 0.3 * total_area:
        total_pixels += hist[threshold]
        threshold -= 1

    mask = cv2.threshold(white, threshold, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(mask, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

    id_contours = []
    movelist_contours = []

    for cnt in contours:
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        if len(poly) == 4 and cv2.isContourConvex(poly):
            area = cv2.contourArea(poly)
            if area > 0.04 * total_area:  # ~101000 pixels
                movelist_contours.append(poly)
            elif area > 0.02 * total_area:  # ~60929 pixels
                id_contours.append(poly)

    return id_contours, movelist_contours


def _prepare_id_image(screen, id_cnt):
    crop = crop_fixed_perspective(screen, id_cnt)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.threshold(blur, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return mask


def _extract_text(image):
    """Extract a team ID from the image"""
    raw = image_to_string(image, config=TESSERACT_CONFIG).strip()

    return raw


def _get_pokemon_info_contour(movelist_contour):
    """Return the pokemon info contour adjacent to the given movelist contour"""
    points = order_points(movelist_contour)
    top_left = points[0] - 1.07 * (points[1] - points[0])
    bottom_left = points[3] - 1.07 * (points[2] - points[3])

    new_points = [
        [top_left],
        [[points[0][0] - 1, points[0][1]]],
        [[points[3][0] - 1, points[3][1]]],
        [bottom_left],
    ]

    return np.array(new_points, dtype=np.int32)


def _prepare_pokemon_info_image(screen, info_contour):
    crop = crop_fixed_perspective(screen, info_contour)
    height, width = crop.shape[:2]
    crop = crop[int(0.4*height):height, 0:int(0.95*width)]

    white = np.min(crop, axis=2)
    mask = cv2.threshold(
        white, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    return mask


def _extract_pokemon_info(pokemon_info_image):
    height, width = pokemon_info_image.shape[:2]

    line_height = int(0.33 * height)
    lines = []
    for i in range(3):
        cropped_line = pokemon_info_image[i*line_height:(i+1)*line_height,
                                          0:width]
        lines.append(
            image_to_string(cropped_line, config=TESSERACT_CONFIG).strip())

    return lines[0], lines[1], lines[2]


def _prepare_movelist_image(screen, movelist_contour):
    crop = crop_fixed_perspective(screen, movelist_contour)
    height, width = crop.shape[:2]
    crop = crop[0:height, int(0.16 * width):width]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return mask


def _extract_movelist(movelist_image):
    height, width = movelist_image.shape[:2]
    line_height = height // 4

    movelist = []
    for i in range(4):
        cropped_line = movelist_image[i*line_height:(i+1)*line_height, 0:width]
        move = image_to_string(cropped_line, config=TESSERACT_CONFIG).strip()
        if move:
            movelist.append(move)

    return movelist


def extract_data_from_image(image: Image):
    debug_images = []
    debug_images.append(DebugImage("source", image))

    image_type = _get_image_type(image)
    logger.info("Image type detected as: %s", image_type.value)

    screen = None
    if image_type == ImageType.PHOTO:
        screen_contour = _get_screen_contour(image)
        if cv2.contourArea(screen_contour) > 0:
            logger.info("Screen found. Cropping out screen...")
            source_with_screen = image.copy()

            cv2.drawContours(
                source_with_screen,
                [screen_contour],
                0,
                (0, 0, 255),
                2
            )

            debug_images.append(DebugImage(
                "source-with-screen", source_with_screen))

            screen = crop_fixed_perspective(image, screen_contour, (1280, 720))
        else:
            logger.info(
                "Screen not found. Using original image as screen image...")

    if screen is None:
        ratio = max(720/image.shape[0], 1280/image.shape[1])
        dsize = (int(ratio*image.shape[1]), int(ratio*image.shape[0]))
        screen = cv2.resize(image, dsize)

    debug_images.append(DebugImage("screen", screen))

    id_contours, movelist_contours = _find_white_regions(screen)
    logger.info("Found %i ID regions, and %i movelist regions.",
                len(id_contours), len(movelist_contours))

    screen_with_regions = screen.copy()

    cv2.drawContours(
        screen_with_regions,
        id_contours,
        -1,
        (0, 0, 255),
        2
    )

    cv2.drawContours(
        screen_with_regions,
        movelist_contours,
        -1,
        (0, 255, 0),
        2
    )

    debug_images.append(DebugImage("screen-with-regions", screen_with_regions))

    if len(id_contours) != 1 or len(movelist_contours) != 6:
        return Result(debug_images, error="Invalid number of ids/movelists")

    team_id_image = _prepare_id_image(screen, id_contours[0])
    debug_images.append(DebugImage("team-id", team_id_image))

    team_id = _extract_text(team_id_image)
    logger.info("Extracted Team ID: %s", team_id)

    team = {"id": team_id, "pokemon": []}

    movelist_contours.sort(key=lambda m: m[0][0][0] + 10 * m[0][0][1])

    for i, movelist_contour in enumerate(movelist_contours):
        logger.info("Extracting %ith Pokemon:", i+1)
        pokemon_info_contour = _get_pokemon_info_contour(movelist_contour)
        pokemon_info_image = _prepare_pokemon_info_image(
            screen, pokemon_info_contour)
        debug_images.append(DebugImage(f"{i}-info", pokemon_info_image))
        species, ability, item = _extract_pokemon_info(pokemon_info_image)

        logger.info("Extracted species text: %s", species)
        logger.info("Extracted ability text: %s", ability)
        logger.info("Extracted item text: %s", item)

        movelist_image = _prepare_movelist_image(screen, movelist_contour)
        debug_images.append(DebugImage(f"{i}-movelist", movelist_image))
        movelist = _extract_movelist(movelist_image)

        for move in movelist:
            logger.info("Extracted move: %s", move)

        pokemon = {"species": species, "ability": ability,
                   "item": item, "movelist": movelist}

        team["pokemon"].append(pokemon)

    return Result(debug_images, team)
