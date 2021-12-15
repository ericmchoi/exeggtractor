""" scrub.py - "lazy" data scrubbing for exeggtractor extraction results

This module contains the code for scrub_team_data which takes the data returned
from extract_data_from_image and scrubs it lazily using a known list of
possible values.
"""
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).parent / "data"
SPECIES_FILE = "monsname.txt"
ABILITIES_FILE = "tokusei.txt"
ITEMS_FILE = "itemname.txt"
MOVES_FILE = "wazaname.txt"


def _read_data_file(filename):
    with open(DATA_DIR / filename, "r", encoding="utf8") as file:
        return file.read().splitlines()


SPECIES = _read_data_file(SPECIES_FILE)
ABILITIES = _read_data_file(ABILITIES_FILE)
ITEMS = _read_data_file(ITEMS_FILE)
MOVES = _read_data_file(MOVES_FILE)


logger = logging.getLogger(__name__)


def _get_best_match(needle: str, haystack: List[str]):
    matcher = SequenceMatcher()
    matcher.set_seq2(needle.lower())
    best_ratio = 0
    best_match = None
    for idx, val in enumerate(haystack):
        matcher.set_seq1(val.lower())
        if matcher.real_quick_ratio() > best_ratio and \
           matcher.ratio() > best_ratio:
            best_ratio = matcher.ratio()
            best_match = idx

    if best_match is None:
        return "", 0

    return haystack[best_match], best_ratio


def _scrub_team_id(raw_id):
    team_id = ""
    match = re.match(r"Team ID (?P<team_id>\w{4} \w{4} \w{4} \w{2})", raw_id)

    if match:
        team_id = match.group("team_id").replace(" ", "")
        team_id = team_id.translate(str.maketrans("oOlIZ", "00112"))

    return team_id


def scrub_team_data(team):
    """Scrubs raw team data extracted from extract_data_from_image

    expects a dictionary of the form:
    {
        "id": str,
        "pokemon":
    }
    """
    scrubbed_id = _scrub_team_id(team["id"])

    scrubbed_pokemon = []
    for mon in team["pokemon"]:
        scrubbed_species, score = _get_best_match(mon["species"], SPECIES)
        if scrubbed_species:
            logger.info("Matched \"%s\" to \"%s\" with score %.2f",
                        mon["species"], scrubbed_species, score)

        scrubbed_ability, score = _get_best_match(mon["ability"], ABILITIES)
        if scrubbed_ability:
            logger.info("Matched \"%s\" to \"%s\" with score %.2f",
                        mon["ability"], scrubbed_ability, score)

        scrubbed_item, score = _get_best_match(mon["item"], ITEMS)
        if scrubbed_item:
            logger.info("Matched \"%s\" to \"%s\" with score %.2f",
                        mon["item"], scrubbed_item, score)

        scrubbed_movelist = []
        for move in mon["movelist"]:
            scrubbed_move, score = _get_best_match(move, MOVES)

            if scrubbed_move:
                logger.info("Matched \"%s\" to \"%s\" with score %.2f",
                            move, scrubbed_move, score)

            scrubbed_movelist.append(scrubbed_move)

        scrubbed_pokemon.append({
            "species": scrubbed_species,
            "ability": scrubbed_ability,
            "item": scrubbed_item,
            "movelist": scrubbed_movelist
        })

    return {"id": scrubbed_id, "pokemon": scrubbed_pokemon}
