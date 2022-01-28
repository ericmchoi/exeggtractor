# Exeggtractor
A Pokemon Sword/Shield screenshot analyzer that extracts Pokemon team data.

![Example](https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/example.png?raw=true)

## Motivation
Pokemon Sword/Shield provides an easy way for players to share Pokemon rental teams for other players to use via an auto-generated ID in game. Players usually share their teams by uploading screenshots of these generated IDs.

However, there isn't a publicly accessible way to grab the data associated with these team IDs, via an API or otherwise. Furthermore, sharing screenshots is a cumbersome, multi-step process for Nintendo switch games, and users may find it more convenient to take a photo of their screen rather than share a clean screenshot.

This program was written to extract Pokemon rental team information from screenshots and from screen photos with certain limitations.

## Usage
### Prerequisites
[Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed on the system for the program to work. For Debian-based OS's, you might run:
```bash
sudo apt install tesseract-ocr
```
Then you can install Exeggtractor via pip:
```bash
pip install git+https://github.com/ericmchoi/exeggtractor.git
```

### As a command-line tool
```
usage: exeggtract [-h] [-v] [-d] [-o OUTPUT_DIR] image

positional arguments:
  image                 path to image file

options:
  -h, --help            show this help message and exit
  -v, --verbose         enable verbose messages
  -d, --debug_images    set additional debug images to be written
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        specify an output directory for debug images
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
### As a library
```python
import exeggtractor

result = exeggtractor.extract_team_from_file("./filepath/to/image.jpg")
print(result)
```
## Program Overview
The flowchart below shows a high-level overview of the steps Exeggtractor takes to extract pokemon team information. A more in-depth explanation of the process can be found at [details](DETAILS.md).

![Overview](https://github.com/ericmchoi/exeggtractor/blob/assets/doc-images/overview.png?raw=true)