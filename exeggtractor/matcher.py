import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).parent / "data"
SPECIES_FILE = "monsname.txt"
ABILITIES_FILE = "tokusei.txt"
ITEMS_FILE = "itemname.txt"
MOVES_FILE = "wazaname.txt"


class BaseMatcher:
    def match_team_id(self, raw: str):
        return raw, 1

    def match_species(self, raw: str):
        return raw, 1

    def match_ability(self, raw: str, **_):
        return raw, 1

    def match_move(self, raw: str, **_):
        return raw, 1

    def match_item(self, raw: str):
        return raw, 1


class NaiveMatcher(BaseMatcher):
    def __init__(self):
        self._species = self._read_data_file(SPECIES_FILE)
        self._abilities = self._read_data_file(ABILITIES_FILE)
        self._items = self._read_data_file(ITEMS_FILE)
        self._moves = self._read_data_file(MOVES_FILE)

    def _read_data_file(self, filename):
        lines = []
        with open(DATA_DIR / filename, "r", encoding="utf8") as file:
            lines = file.read().splitlines()

        return lines

    def _get_best_match(self, needle: str, haystack: List[str]):
        matcher = SequenceMatcher()
        matcher.set_seq2(needle.lower())
        best_ratio = 0
        best_match = None
        for idx, val in enumerate(haystack):
            matcher.set_seq1(val.lower())
            if matcher.real_quick_ratio() > best_ratio and matcher.ratio() > best_ratio:
                best_ratio = matcher.ratio()
                best_match = idx

        if best_match is None:
            return "", 0

        return haystack[best_match], best_ratio

    def match_team_id(self, raw: str):
        team_id = ""

        match = re.match(r".*(?P<team_id>\w{4} \w{4} \w{4} \w{2})$", raw.strip())

        if match:
            team_id = match.group("team_id").replace(" ", "")
            team_id = team_id.translate(str.maketrans("oOlIZ", "00112"))

        return team_id, int(team_id != "")

    def match_species(self, raw: str):
        match = re.match(r"^(?P<to_match>.*)(?:Lv|Ly|Lw)", raw.strip())
        to_match = match.group("to_match") if match else raw

        return self._get_best_match(to_match, self._species)

    def match_ability(self, raw: str, **_):
        return self._get_best_match(raw, self._abilities)

    def match_move(self, raw: str, **_):
        return self._get_best_match(raw, self._moves)

    def match_item(self, raw: str):
        return self._get_best_match(raw, self._items)
