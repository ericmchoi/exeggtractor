""" models.py - data models for exeggtractor

This module defines types and dataclasses that are used in exeggtractor
"""
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import numpy as np

Image = np.ndarray


class ImageType(str, Enum):
    """Enum type for recognized image type"""
    SCREENSHOT = 'screenshot'
    PHOTO = 'photo'


class Pokemon(TypedDict):
    """Class representing a pokemon and its associated data"""
    species: str
    ability: str
    item: str
    movelist: list[str]


class Team(TypedDict):
    """Class representing a pokemon team and its associated data"""
    id: str
    pokemon: list[Pokemon]


@dataclass
class DebugImage:
    """Dataclass used for storing debug image data"""
    name: str
    image: Image


@dataclass
class Result:
    """Dataclass used to represent data returned from the extractor"""
    debug_images: list[DebugImage]
    team: dict = None
    error: str = None
