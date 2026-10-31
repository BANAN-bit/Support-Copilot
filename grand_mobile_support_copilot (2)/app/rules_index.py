from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("grand_mobile_copilot.rules")

DEFAULT_RULES_SEARCH_PATHS: Tuple[str, ...] = (
    "data/rules.json",
    "rules.json",
    "handout/rules.json",
    "../data/rules.json",
)


def resolve_default_rules_path() -> str:
    """
    If RULES_PATH is not set in the environment, search for rules.json in order:
    data/rules.json, rules.json, handout/rules.json, ../data/rules.json.
    """
    env_path = os.getenv("RULES_PATH")
    if env_path:
        return env_path
    for candidate in DEFAULT_RULES_SEARCH_PATHS:
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_RULES_SEARCH_PATHS[0]


# Matches numbered rule start at beginning of line: "1.1 ...", "1.1.2 ...", "5.4.1 ...", "ЖБ 1.2.7 ...", "GZ 1.2 ..."
RULE_START_REGEX = re.compile(
    r"^\s*(?P<id>(?:(?:ЖБ|GZ)\s+)?\d+(?:\.\d+)+)\s*[\.\—\-\:]?\s*(?P<text>.*)\s*$",
    re.UNICODE,
)

# Matches Roman numeral sub-section headers like "I. ОБЩИЕ ПОЛОЖЕНИЯ:", "II. ПРАВИЛА ОБЩЕНИЯ..."
ROMAN_HEADER_REGEX = re.compile(
    r"^\s*(?:[IVX]+)\.\s*[^\d\n]+$",
    re.UNICODE,
)

# Only sections 30, 31, and 32 are indexed for citation (confirmed by taxonomy.json:
# bare numbers 1.1..5.9 for id=30, "ЖБ " for id=31, and "GZ " / "GZ" for id=32).
# Sections 33..43 do not have official citation prefixes in rules.json or taxonomy.json.
SECTION_ID_PREFIX: Dict[int, str] = {
    30: "",      # Общие правила игры -> "1.1".."5.9" (41 items)
    31: "ЖБ ",   # Правила жалоб -> "ЖБ 1.1".."ЖБ 1.5.2" (26 items)
    32: "GZ ",   # Правила нахождения в Зелёной Зоне -> "GZ 1.1".."GZ 2.3" (8 items) + "GZ"
}


@dataclass
class RuleItem:
    id: str
    text: str
    section_id: int
    section_title: str
    raw_ref: str = ""

    def format_for_prompt(self) -> str:
        return f"[{self.id}] {self.text}"


class RulesIndex:
    """
    Parses data/rules.json (11 sections).
    Indexes for citation ONLY sections 30 ("Общие правила игры"), 31 ("Правила жалоб" with "ЖБ " prefix),
    and 32 ("Правила нахождения в Зелёной Зоне" with "GZ " prefix + "GZ"), as confirmed by taxonomy.json.
    If rules.json is missing, logs a clear message and sets self.load_error (degraded mode) instead of crashing.
    """

    def __init__(self, rules_path: Optional[str] = None) -> None:
        if rules_path is None:
            rules_path = resolve_default_rules_path()
        self.rules_path = rules_path
        self.sections: List[dict] = []
        self.index: Dict[str, RuleItem] = {}
        self.section_counts: Dict[int, int] = {}
        self.load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.rules_path):
            self.load_error = f"rules.json not found at {self.rules_path}; mount ./data into /app/data"
            logger.error(self.load_error)
            self.sections = []
            self.index.clear()
            self.section_counts.clear()
            return

        self.load_error = None
        with open(self.rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.sections = data.get("rules") or data.get("sections") or []
        self.index.clear()
        self.section_counts.clear()

        for sec in self.sections:
            sec_id = int(sec.get("id", sec.get("section_id", 0)))
            sec_title = str(sec.get("category", sec.get("title", "")))
            raw_text = str(sec.get("text", ""))

            # Parse raw numbered points in every section for auditing section_counts
            raw_parsed = self._parse_section_text(raw_text, sec_id, sec_title, prefix="")
            self.section_counts[sec_id] = len(raw_parsed)

            # Index for citation only sections 30, 31, 32 (confirmed by taxonomy.json)
            if sec_id not in SECTION_ID_PREFIX:
                continue

            prefix = SECTION_ID_PREFIX[sec_id]
            citable_items = self._parse_section_text(raw_text, sec_id, sec_title, prefix=prefix)
            for item in citable_items:
                self.index[item.id] = item

            if prefix == "GZ " and "GZ" not in self.index:
                gz_1_1 = self.index.get("GZ 1.1")
                gz_1_2 = self.index.get("GZ 1.2")
                summary_parts = []
                if gz_1_1:
                    summary_parts.append(f"GZ 1.1: {gz_1_1.text}")
                if gz_1_2:
                    summary_parts.append(f"GZ 1.2: {gz_1_2.text}")
                gz_text = (
                    " ".join(summary_parts)
                    if summary_parts
                    else "Запрещены убийства, нанесение урона и криминальные действия в Зелёной Зоне (Green Zone | GZ)."
                )
                self.index["GZ"] = RuleItem(
                    id="GZ",
                    text=gz_text,
                    section_id=sec_id,
                    section_title=sec_title,
                    raw_ref="GZ",
                )

    def _parse_section_text(
        self,
        raw_text: str,
        sec_id: int,
        sec_title: str,
        prefix: str,
    ) -> List[RuleItem]:
        items: List[RuleItem] = []
        current_id: Optional[str] = None
        current_raw_ref: str = ""
        current_lines: List[str] = []

        def _flush() -> None:
            nonlocal current_id, current_raw_ref, current_lines
            if current_id is not None:
                body = " ".join(line.strip() for line in current_lines if line.strip()).strip()
                if body:
                    items.append(
                        RuleItem(
                            id=current_id,
                            text=body,
                            section_id=sec_id,
                            section_title=sec_title,
                            raw_ref=current_raw_ref,
                        )
                    )
            current_id = None
            current_raw_ref = ""
            current_lines = []

        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if ROMAN_HEADER_REGEX.match(stripped):
                continue

            m_num = RULE_START_REGEX.match(stripped)
            if m_num:
                _flush()
                raw_ref = re.sub(r"\s+", " ", m_num.group("id").strip())
                if raw_ref.upper().startswith("ЖБ "):
                    norm_id = "ЖБ " + raw_ref.split(" ", 1)[1]
                elif raw_ref.upper().startswith("GZ "):
                    norm_id = "GZ " + raw_ref.split(" ", 1)[1]
                else:
                    norm_id = f"{prefix}{raw_ref}".strip()
                current_id = norm_id
                current_raw_ref = raw_ref
                first_txt = m_num.group("text").strip()
                if first_txt:
                    current_lines.append(first_txt)
                continue

            if current_id is not None:
                if stripped.startswith("[All]") or stripped.startswith("[Пра-во] -"):
                    continue
                current_lines.append(stripped)

        _flush()
        return items

    def get(self, rule_id: str) -> Optional[RuleItem]:
        return self.index.get(rule_id)

    def exists(self, rule_id: str) -> bool:
        return rule_id in self.index

    def all_items(self) -> List[RuleItem]:
        return list(self.index.values())
