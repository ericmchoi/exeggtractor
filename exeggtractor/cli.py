import logging
import os
from argparse import ArgumentParser
import json

import cv2

from .extract import extract_data_from_image


def main():
    """Handle command line usage of this module"""
    parser = ArgumentParser()
    parser.add_argument("image", help="filepath image")
    parser.add_argument("-v",
                        "--verbose",
                        help="enable verbose messages",
                        action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s")

        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    result = extract_data_from_image(cv2.imread(args.image))

    if result.error:
        print(result.error)
    else:
        print(json.dumps(result.team))

    os.makedirs("temp/", exist_ok=True)
    for image in result.debug_images:
        cv2.imwrite(f'temp/{image.name}.jpg', image.image)


if __name__ == "__main__":
    main()
