"""
helpers.py
~~~~~~~~~
This module implements some wrapper convenience functions
"""
import logging
from urllib.request import urlopen

import cv2
import numpy as np

from .extractor import DebugImageLevel, Extractor

LOGGER = logging.getLogger(__name__)


def extract_team_from_file(
    filename, debug_image_level=DebugImageLevel.NONE, debug_image_dir=None
):
    """Extract a pokemon team's information from a image file"""
    LOGGER.info("Opening file: %s", filename)

    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {filename}")

    extractor = Extractor(
        image, debug_image_level=debug_image_level, debug_image_dir=debug_image_dir
    )
    result = extractor.extract()

    return result


def extract_team_from_url(
    url, debug_image_level=DebugImageLevel.NONE, debug_image_dir=None
):
    """Extract a pokemon team's information from a image url"""
    LOGGER.info("Opening url: %s", url)

    req = urlopen(url)
    image = np.asarray(bytearray(req.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    extractor = Extractor(
        image, debug_image_level=debug_image_level, debug_image_dir=debug_image_dir
    )
    result = extractor.extract()

    return result
