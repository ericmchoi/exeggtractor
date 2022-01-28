"""
extractor.py
~~~~~~~~~~~~~~~~
This module contains the Extractor class which drives the core logic for
extracting data from a Pokemon Rental Team screenshot (or photo)
"""
import logging
from enum import Enum
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from colour import delta_E
from pytesseract import image_to_string

from .matcher import BaseMatcher, NaiveMatcher
from .utils import crop_fixed_perspective, draw_hough_lines, rectangularContour

TESSERACT_CONFIG = r"--psm 7 -c page_separator='' --dpi 72"
CORNER_TESTS = [
    {
        "x": 0,
        "y": 0,
        "expected_color": [86.58447, -45.359375, 0.84375],
        "threshold": 10,
    },
    {
        "x": 0,
        "y": -1,
        "expected_color": [71.25244, -22.265625, -34.046875],
        "threshold": 10,
    },
    {"x": -1, "y": 0, "expected_color": [0, 0, 0], "threshold": 2},
    {"x": -1, "y": -1, "expected_color": [0, 0, 0], "threshold": 2},
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Image = np.ndarray


class State(Enum):
    """Enum type for extractor status"""

    READY = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4


class DebugImageLevel(Enum):
    """Enum type for debug image levels"""

    NONE = 1
    BASIC = 2
    ALL = 3

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class ImageType(str, Enum):
    """Enum type for recognized image type"""

    SCREENSHOT = "screenshot"
    PHOTO = "photo"


class ExtractionError(Exception):
    """Exception for generic errors during team extraction"""


class ExtractorNotReady(ExtractionError):
    """Exception for when the Extractor is not ready"""


class ScreenNotFound(ExtractionError):
    """Exception for when the Extractor cannot find the screen in the image"""


class Extractor:
    """Class that handles the extraction of Pokemon team data from an image"""

    def __init__(
        self,
        image: Image,
        matcher: BaseMatcher | None = None,
        debug_image_level: DebugImageLevel | None = None,
        debug_image_dir: str | None = None,
    ):
        self.source = image

        self.debug_image_level = DebugImageLevel.NONE
        if debug_image_level is not None:
            self.debug_image_level = debug_image_level

        self.debug_image_dir = None
        if debug_image_dir is not None:
            self.debug_image_dir = Path(debug_image_dir)

        if matcher is None:
            self.matcher = NaiveMatcher()
        else:
            self.matcher = matcher

        self.status = State.READY
        self.debug_image_count = 0
        self.result = None
        self.raw = None

    def _save_image(self, label, image, level=DebugImageLevel.BASIC):
        if self.debug_image_dir is not None and level <= self.debug_image_level:
            filename = str(
                self.debug_image_dir / f"{self.debug_image_count:02}-{label}.jpg"
            )
            cv2.imwrite(filename, image)
            self.debug_image_count += 1

    def _detect_image_type(self):
        if self.source.shape[0:2] not in [(720, 1280), (1080, 1920)]:
            return ImageType.PHOTO

        for test in CORNER_TESTS:
            bgr = self.source[test["x"], test["y"]]

            # convert bgr values so that cvtColor returns values in the correct range
            bgr = bgr.reshape((1, 1, 3)).astype(np.float32) / 255

            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

            color_diff = delta_E(lab, np.array([[test["expected_color"]]]))
            if color_diff > test["threshold"]:
                return ImageType.PHOTO

        return ImageType.SCREENSHOT

    def _quantize_colors(self, image, color_count):
        samples = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, colors = cv2.kmeans(
            samples, color_count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        color_quantized = colors[labels.flatten()].reshape(image.shape).astype(np.uint8)

        return labels, colors, color_quantized

    def _find_screen_kmeans(self, image):
        height, width = image.shape[:2]

        color_count = 3
        color_labels, _, color_quantized = self._quantize_colors(image, color_count)
        self._save_image("source-quantized", color_quantized, DebugImageLevel.ALL)

        potential_screens = []
        components_image = np.zeros(image.shape, dtype=np.uint8)
        for color in range(color_count):
            color_mask = (
                ((color_labels == color) * 255)
                .reshape((height, width, 1))
                .astype(np.uint8)
            )
            self._save_image(f"color-mask-{color}", color_mask, DebugImageLevel.ALL)

            # connected component analysis
            component_count, components, stats, _ = cv2.connectedComponentsWithStats(
                color_mask
            )

            for component in range(1, component_count):
                component_left = stats[component, cv2.CC_STAT_LEFT]
                component_top = stats[component, cv2.CC_STAT_TOP]
                component_width = stats[component, cv2.CC_STAT_WIDTH]
                component_height = stats[component, cv2.CC_STAT_HEIGHT]

                # reject components that do not take up at least half the photo's width
                if component_width < 0.50 * width:
                    continue

                # reject components that touch the image edge
                if (
                    component_left == 0
                    or component_top == 0
                    or component_left + component_width >= width
                    or component_top + component_height >= height
                ):
                    continue

                component_mask = ((components == component) * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for cnt in contours:
                    eps = 0.01 * cv2.arcLength(cnt, True)
                    poly = cv2.approxPolyDP(cnt, eps, True)
                    if len(poly) == 4 and cv2.isContourConvex(poly):
                        # fill the connected component with a random color
                        color = np.random.randint(0, 255, 3)
                        components_image[components == component] = color
                        cv2.rectangle(
                            components_image,
                            (component_left, component_top),
                            (
                                component_left + component_width,
                                component_top + component_height,
                            ),
                            (0, 0, 255),
                            max(width, height) // 200,
                        )

                        bounding_area = component_width * component_height
                        diff = abs(cv2.contourArea(poly) - bounding_area)
                        potential_screens.append((poly, diff))

        self._save_image("connected-components", components_image, DebugImageLevel.ALL)

        if len(potential_screens) == 0:
            raise ScreenNotFound("Unable to find screen")

        # choose the contour that's most similar to it's bounding box
        # i.e. the "straightest" rectangle
        screen, _ = min(potential_screens, key=lambda x: x[1])
        return screen

    def _find_screen_canny(self, image):
        # preprocess for contour finding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)

        # calculate canny thresholds from otsu's
        high_threshold, _ = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low_threshold = 0.5 * high_threshold

        canny = cv2.Canny(blur, low_threshold, high_threshold)
        self._save_image("canny", canny)

        # find contours
        contours, _ = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        potential_screens = []
        for cnt in contours:
            _, _, width, _ = cv2.boundingRect(cnt)
            if width < 0.5 * image.shape[1]:
                continue

            eps = 0.01 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, eps, True)
            poly_area = cv2.contourArea(poly)
            if len(poly) == 4 and cv2.isContourConvex(poly):
                potential_screens.append((poly, poly_area))

        if len(potential_screens) == 0:
            raise ScreenNotFound("Unable to find screen")

        # choose the contour that's most similar to it's approximated polygon
        screen, _ = min(potential_screens, key=lambda x: x[1])

        return screen

    def _get_screen(self):
        height, width = self.source.shape[:2]
        ratio = max(720 / height, 1280 / width)
        dsize = (
            int(ratio * width),
            int(ratio * height),
        )
        resized = cv2.resize(self.source, dsize)

        screen_contour = np.int32(self._find_screen_kmeans(resized) / ratio)

        source_with_screen = self.source.copy()
        cv2.drawContours(
            source_with_screen,
            [screen_contour],
            0,
            (0, 0, 255),
            max(height, width) // 200,
        )
        self._save_image("source-with-screen", source_with_screen)

        screen = crop_fixed_perspective(self.source, screen_contour, (1280, 690))

        # add the black bar that is usually omitted by the screen finding techniques
        screen.resize((720, 1280, 3))

        return screen

    def _get_white_regions(self, image):
        height, width = image.shape[:2]

        color_count = 3
        color_labels, colors, color_quantized = self._quantize_colors(
            image, color_count
        )
        self._save_image("screen-quantized", color_quantized, DebugImageLevel.ALL)

        # color analysis to find the whitest cluster
        color_diffs = []
        for color in colors:
            bgr = color.reshape((1, 1, 3)) / 255
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
            diff = delta_E(lab, np.array([[[100, 0, 0]]], dtype=np.float32))

            color_diffs.append(diff)

        whitest = np.argmin(color_diffs)
        mask = (
            ((color_labels == whitest) * 255)
            .reshape((height, width, 1))
            .astype(np.uint8)
        )
        self._save_image("white-mask", mask, DebugImageLevel.ALL)

        # clean up vertical edges
        opening_x = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25, 5), np.uint8))
        self._save_image("opening-x", opening_x, DebugImageLevel.ALL)

        # preprocessing for hough lines
        sobel_x = cv2.Sobel(opening_x, cv2.CV_64F, 1, 0)
        sobel_x = np.absolute(sobel_x).astype(np.uint8)
        sobel_x = cv2.dilate(sobel_x, np.ones((2, 2), np.uint8))
        self._save_image("sobel-x", sobel_x, DebugImageLevel.ALL)

        lines = cv2.HoughLinesWithAccumulator(sobel_x, 1, np.pi, 400)
        vertical_lines = cv2.cvtColor(sobel_x, cv2.COLOR_GRAY2BGR)
        draw_hough_lines(vertical_lines, lines, (0, 0, 255), 2)
        self._save_image("vertical-hough-lines", vertical_lines, DebugImageLevel.ALL)

        if len(lines) < 4:
            raise ExtractionError("Could not find enough vertical lines.")

        # merging similar hough lines
        samples = np.repeat(lines[:, 0, 0], lines[:, 0, 2].astype(np.int32))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, verticals = cv2.kmeans(
            samples, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # clean up horizontal edges
        opening_y = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 15), np.uint8))
        self._save_image("opening-y", opening_y, DebugImageLevel.ALL)

        # preprocessing for hough lines
        sobel_y = cv2.Sobel(opening_y, cv2.CV_64F, 0, 1)
        sobel_y = np.absolute(sobel_y).astype(np.uint8)
        sobel_y = cv2.dilate(sobel_y, np.ones((2, 2), np.uint8))
        self._save_image("sobel-y", sobel_y, DebugImageLevel.ALL)

        # use half the image here to reduce noise
        lines = cv2.HoughLinesWithAccumulator(
            sobel_y[:, : width // 2], 1, np.pi / 2, 200
        )

        # try finding the horizontal lines from the other half
        if len(lines) < 8:
            lines = cv2.HoughLinesWithAccumulator(
                sobel_y[:, width // 2 :], 1, np.pi / 2, 200
            )

        horizontal_lines = cv2.cvtColor(sobel_y, cv2.COLOR_GRAY2BGR)
        draw_hough_lines(horizontal_lines, lines, (0, 0, 255), 2)
        self._save_image(
            "horizontal-hough-lines", horizontal_lines, DebugImageLevel.ALL
        )

        if len(lines) < 8:
            raise ExtractionError("Could not find enough horizontal lines.")

        # merging similar hough lines
        samples = np.repeat(lines[:, 0, 0], lines[:, 0, 2].astype(np.int32))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, horizontals = cv2.kmeans(
            samples, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        verticals = verticals.flatten().astype(np.int32)
        horizontals = horizontals.flatten().astype(np.int32)
        verticals.sort()
        horizontals.sort()

        merged_lines = image.copy()
        for x in verticals:  # pylint: disable=invalid-name
            cv2.line(merged_lines, (x, 0), (x, height), (255, 0, 255), 1, cv2.LINE_AA)
        for y in horizontals:  # pylint: disable=invalid-name
            cv2.line(merged_lines, (0, y), (width, y), (0, 0, 255), 1, cv2.LINE_AA)
        self._save_image("merged-lines", merged_lines, DebugImageLevel.ALL)

        movelist_contours = [
            rectangularContour(
                verticals[0], verticals[1], horizontals[0], horizontals[1]
            ),
            rectangularContour(
                verticals[2], verticals[3], horizontals[0], horizontals[1]
            ),
            rectangularContour(
                verticals[0], verticals[1], horizontals[2], horizontals[3]
            ),
            rectangularContour(
                verticals[2], verticals[3], horizontals[2], horizontals[3]
            ),
            rectangularContour(
                verticals[0], verticals[1], horizontals[4], horizontals[5]
            ),
            rectangularContour(
                verticals[2], verticals[3], horizontals[4], horizontals[5]
            ),
        ]

        id_contour = rectangularContour(
            verticals[0], verticals[2], horizontals[6], horizontals[7]
        )

        return movelist_contours, id_contour

    def _extract_text(
        self, image, match_function, white_text=False, threshold=0.9, retries=3
    ):
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        attempt_images = []

        best_match = None
        best_score = 0
        matched_raw = None

        for i in range(retries):
            logger.info("Extracting text, attempt %i", 1 + i)

            # preprocess for OCR
            values = gray.reshape([gray.shape[0] * gray.shape[1], -1]).astype(
                np.float32
            )
            criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
            _, labels, centers = cv2.kmeans(values, 2 + i, None, criteria, 10, 0)

            if white_text:
                mask = np.where(labels == np.argmax(centers), 255, 0)
            else:
                mask = np.where(labels == np.argmin(centers), 255, 0)

            mask = mask.reshape(gray.shape).astype(np.uint8)

            # connected component analysis
            num_components, components, stats, _ = cv2.connectedComponentsWithStats(
                mask
            )

            cleaned_mask = np.zeros(mask.shape, mask.dtype)
            for component in range(1, num_components):
                component_left = stats[component, cv2.CC_STAT_LEFT]
                component_top = stats[component, cv2.CC_STAT_TOP]
                component_width = stats[component, cv2.CC_STAT_WIDTH]
                component_height = stats[component, cv2.CC_STAT_HEIGHT]

                # reject components that touch the image edge
                if (
                    component_left == 0
                    or component_top == 0
                    or component_left + component_width >= width
                    or component_top + component_height >= height
                ):
                    continue

                cleaned_mask[components == component] = 255
            cleaned_mask = 255 - cleaned_mask
            attempt_images.append(cleaned_mask)

            raw = image_to_string(cleaned_mask, config=TESSERACT_CONFIG).strip()
            logger.info("Extracted text: %s", raw)

            if match_function is None:
                break

            matched, score = match_function(raw)
            logger.info('Matched text to "%s" with score %.2f', matched, score)

            if score >= threshold:
                return matched, raw, attempt_images

            if score > best_score:
                best_match, best_score, matched_raw = matched, score, raw

        return best_match, matched_raw, attempt_images

    def _get_info_contours(self, movelist):
        # pylint: disable=invalid-name
        x1 = movelist[0][0][0] - 1.06 * (movelist[1][0][0] - movelist[0][0][0])
        x2 = movelist[0][0][0]

        y0 = movelist[0][0][1]
        height = movelist[2][0][1] - y0

        species = rectangularContour(x1, x2, y0 + 0.4 * height, y0 + 0.6 * height)
        ability = rectangularContour(x1, x2, y0 + 0.6 * height, y0 + 0.8 * height)
        item = rectangularContour(x1, x2, y0 + 0.8 * height, y0 + height)

        return species, ability, item

    def _get_move_contours(self, movelist):
        # pylint: disable=invalid-name
        x1 = movelist[0][0][0] + 0.16 * (movelist[1][0][0] - movelist[0][0][0])
        x2 = movelist[1][0][0]

        y0 = movelist[0][0][1]
        height = movelist[2][0][1] - y0

        moves = [
            rectangularContour(
                x1, x2, y0 + 0.25 * i * height, y0 + 0.25 * (i + 1) * height
            )
            for i in range(4)
        ]

        return moves

    def extract(self):
        """extract pokemon team data from an image"""
        if self.status is not State.READY:
            raise ExtractorNotReady("Extractor is already running or done running.")

        self.status = State.RUNNING
        self._save_image("source", self.source)

        image_type = self._detect_image_type()
        logger.info("Image type detected as: %s", image_type.value)

        screen = None
        if image_type == ImageType.PHOTO:
            try:
                screen = self._get_screen()
                logger.info("Found screen in photo.")
            except ScreenNotFound:
                logger.info(
                    "Unable to find screen. Using original image as screen image."
                )

        if screen is None:
            ratio = max(720 / self.source.shape[0], 1280 / self.source.shape[1])
            dsize = (
                int(ratio * self.source.shape[1]),
                int(ratio * self.source.shape[0]),
            )
            screen = cv2.resize(self.source, dsize)

        self._save_image("screen", screen)

        movelist_contours, id_contour = self._get_white_regions(screen)
        white_regions = screen.copy()
        cv2.drawContours(white_regions, [id_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(white_regions, movelist_contours, -1, (0, 255, 255), 2)
        self._save_image("white-regions", white_regions)

        logger.info("Extracting team ID...")
        team_id_image = crop_fixed_perspective(screen, id_contour)
        team_id, raw_id, attempts = self._extract_text(
            team_id_image, self.matcher.match_team_id, False
        )

        for attempt, attempt_image in enumerate(attempts):
            self._save_image(f"team-id-{1+attempt}", attempt_image, DebugImageLevel.ALL)

        text_regions = screen.copy()
        pokemon = []
        raw_pokemon = []
        for slot, movelist_contour in enumerate(movelist_contours):
            logger.info("Extracting Pokemon %i...", 1 + slot)
            species_contour, ability_contour, item_contour = self._get_info_contours(
                movelist_contour
            )

            logger.info("Extracting species...")
            species_img = crop_fixed_perspective(screen, species_contour)
            species, raw_species, species_attempts = self._extract_text(
                species_img, self.matcher.match_species, True
            )

            for attempt, attempt_image in enumerate(species_attempts):
                self._save_image(
                    f"{1+slot}-species-{1+attempt}", attempt_image, DebugImageLevel.ALL
                )

            logger.info("Extracting ability...")
            ability_img = crop_fixed_perspective(screen, ability_contour)
            ability, raw_ability, ability_attempts = self._extract_text(
                ability_img, partial(self.matcher.match_ability, species=species), True
            )

            for attempt, attempt_image in enumerate(ability_attempts):
                self._save_image(
                    f"{1+slot}-ability-{1+attempt}", attempt_image, DebugImageLevel.ALL
                )

            logger.info("Extracting item...")
            item_img = crop_fixed_perspective(screen, item_contour)
            item, raw_item, item_attempts = self._extract_text(
                item_img, self.matcher.match_item, True
            )

            for attempt, attempt_image in enumerate(item_attempts):
                self._save_image(
                    f"{1+slot}-item-{1+attempt}", attempt_image, DebugImageLevel.ALL
                )

            move_contours = self._get_move_contours(movelist_contour)

            movelist = []
            raw_movelist = []
            for move_slot, move_contour in enumerate(move_contours):
                logger.info("Extracting move %i...", 1 + move_slot)
                move_img = crop_fixed_perspective(screen, move_contour)
                move, raw_move, move_attempts = self._extract_text(
                    move_img, partial(self.matcher.match_move, species=species)
                )

                for attempt, attempt_image in enumerate(move_attempts):
                    self._save_image(
                        f"{1+slot}-move-{1+move_slot}-{1+attempt}",
                        attempt_image,
                        DebugImageLevel.ALL,
                    )

                movelist.append(move)
                raw_movelist.append(raw_move)

            mon = {
                "species": species,
                "ability": ability,
                "item": item,
                "movelist": movelist,
            }

            raw_mon = {
                "species": raw_species,
                "ability": raw_ability,
                "item": raw_item,
                "movelist": raw_movelist,
            }

            info_contours = [species_contour, ability_contour, item_contour]
            cv2.drawContours(text_regions, info_contours, -1, (0, 0, 255), 2)
            cv2.drawContours(text_regions, move_contours, -1, (0, 0, 255), 2)

            pokemon.append(mon)
            raw_pokemon.append(raw_mon)

        self._save_image("text-regions", text_regions)
        self.result = {"id": team_id, "pokemon": pokemon}
        self.raw = {"id": raw_id, "pokemon": raw_pokemon}

        return self.result
