""" cli.py - command line script for exeggtractor

This module contains the code for command line usage of the extractor library.
The script is invoked as follows:

usage: exeggtract [-h] [-v] [-o OUTPUT_DIR] [-r] image

positional arguments:
  image                 path to image file

options:
  -h, --help            show this help message and exit
  -v, --verbose         enables verbose messages
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        specifies an output directory for debug images
  -r, --raw             disables data scrubbing and returns raw data
"""

import json
import logging
import os
import pathlib
import sys
from argparse import ArgumentParser

import cv2

from .extract import extract_data_from_image
from .scrub import scrub_team_data


def main():
    """Handle command line usage of this module"""
    parser = ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("-v",
                        "--verbose",
                        help="enables verbose messages",
                        action="store_true")
    parser.add_argument("-o",
                        "--output_dir",
                        help="specifies an output directory for debug images")
    parser.add_argument("-r",
                        "--raw",
                        help="disables data scrubbing and returns raw data",
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

    image = cv2.imread(args.image)

    if image is None:
        print(f"Could not read image at: {args.image}", file=sys.stderr)
        sys.exit(os.EX_NOINPUT)

    result = extract_data_from_image(image)

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)

        for image in result.debug_images:
            cv2.imwrite(str(output_dir / f'{image.name}.jpg'), image.image)

    if result.error:
        print(result.error)
    else:
        if args.raw:
            print(json.dumps(result.team))
        else:
            scrubbed_team = scrub_team_data(result.team)
            print(json.dumps(scrubbed_team))


if __name__ == "__main__":
    main()
