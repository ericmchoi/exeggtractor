# Exeggtractor
A Pokemon Sword/Shield screenshot analyzer that extracts Pokemon team data.

![Example]()

## Motivation
Pokemon Sword/Shield provides an easy way for players to share Pokemon rental teams for other players to use via an auto-generated ID in game. Players usually share their teams by generating these IDs and sharing screenshots of them to others.

However, there isn't a publicly accessible way to grab the data associated with these team IDs, via an API or otherwise. Furthermore, sharing screenshots is a cumbersome, multi-step process for Nintendo switch games, and users may find it more convenient to take a photo of their screen rather than share a clean screenshot.

This program was written to extract Pokemon rental team information from both screenshots and screen photos.

## Usage
### Prerequisites
[Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed on the system for the program to work. For Debian-based OS's, you might run:
```bash
sudo apt install tesseract-ocr
```
### Installation
```bash
git clone https://github.com/ericmchoi/exeggtractor.git
cd exeggtractor
pip install
```
### Usage
Exeggtractor can be used both as a command-line tool and a library.
#### Command-line
```
usage: exeggtract [-h] [-v] [-o OUTPUT_DIR] [-d] image

positional arguments:
  image                 path to image file

options:
  -h, --help            show this help message and exit
  -v, --verbose         enables verbose messages
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        specifies an output directory for debug images
  -d, --debug_images    sets all debug images to be written
```
##### Example
```
$ exeggtract test.jpg 
{
  "id": "00000000M1PRJN",
  "pokemon": [
    {
      "species": "Tyranitar",
      "ability": "Sand Stream",
      "item": "Life Orb",
      "movelist": [
        "Rock Slide",
        "Crunch",
        "Protect",
        "Low Kick"
      ]
    },
    {
      "species": "Excadrill",
      "ability": "Sand Rush",
      "item": "Focus Sash",
      "movelist": [
        "Rock Slide",
        "High Horsepower",
        "Protect",
        "Swords Dance"
      ]
    },
    {
      "species": "Braviary",
      "ability": "Defiant",
      "item": "Choice Scarf",
      "movelist": [
        "Tailwind", 
        "U-turn",
        "Brave Bird",
        "Close Combat"
      ]
    },
    {
      "species": "Hydreigon",
      "ability": "Levitate",
      "item": "Choice Specs",
      "movelist": [
        "Dragon Pulse",
        "Dark Pulse",
        "Flamethrower",
        "Draco Meteor"
      ]
    },
    {
      "species": "Rotom",
      "ability": "Levitate",
      "item": "Sitrus Berry",
      "movelist": [
        "Thunderbolt",
        "Hydro Pump",
        "Volt Switch",
        "Protect"
      ]
    },
    {
      "species": "Mimikyu",
      "ability": "Disguise",
      "item": "Lum Berry",
      "movelist": [
        "Play Rough",
        "Shadow Sneak",
        "Taunt",
        "Swords Dance"
      ]
    }
  ]
}
```
#### Library
```python
import cv2
from extractor import Extractor


```