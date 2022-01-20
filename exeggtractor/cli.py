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
import sys
import os
from argparse import ArgumentParser

from .api import extract_team_from_file
from .extractor import DebugImageLevel


def main():
    """Handle command line usage of this module"""
    parser = ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument(
        "-v", "--verbose", help="enable verbose messages", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--debug_images",
        help="set additional debug images to be written",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--output_dir", help="specify an output directory for debug images"
    )
    args = parser.parse_args()

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_format = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    debug_image_level = None
    if args.debug_images:
        debug_image_level = DebugImageLevel.ALL

    try:
        result = extract_team_from_file(args.image, debug_image_level, args.output_dir)
    except FileNotFoundError:
        print(f"Could not read image at: {args.image}", file=sys.stderr)
        sys.exit(os.EX_NOINPUT)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
