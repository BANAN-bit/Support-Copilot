import re
import unicodedata
from typing import Any, List, Optional

# Strictly Kazakh-only mojibake characters in cp1251 (excluding Ukrainian є, і, ї unless Kazakh context is present)
CP1251_STRICT_KAZAKH_CHARS = set("ќЌѕЅјЈўЎѓЃ")

# Full mapping for cp1251/legacy mojibake of Kazakh letters:
# ќ→қ, ґ/ѓ→ғ, ѕ→ң, є→ә, ј→ө, ў→ұ, ї→ү (and uppercase counterparts)
CP1251_KAZAKH_MAP = str.maketrans({
    "ќ": "қ",
    "Ќ": "Қ",
    "ґ": "ғ",
    "Ґ": "Ғ",
    "ѓ": "ғ",
    "Ѓ": "Ғ",
    "ѕ": "ң",
    "Ѕ": "Ң",
    "є": "ә",
    "Є": "Ә",
    "ј": "ө",
    "Ј": "Ө",
    "ў": "ұ",
    "Ў": "Ұ",
    "ї": "ү",
    "Ї": "Ү",
})

KAZAKH_NATIVE_CHARS = set("қғңәөұүһҚҒҢӘӨҰҮҺ")

UKRAINIAN_MARKER_REGEX = re.compile(
    r"\b(?:щоб|знав|як|потрібно|розмовляти|захищался|захичался|деємшиков|отбьівался|"
    r"памагіте|памагі|поможіть|поможітя|помогітє|переверніть|перевернітє|"
    r"адмін|адміни|адміністратор|адміністратори|будемо|використанні|цікавить|"
    r"зав'язали|розв'язати|потірявся|він|має|в'язати|поліції|купив|вона|"
    r"інвентарю|пошті|немає|нема|евакуатор|визвати|устроєтса)\b",
    re.IGNORECASE,
)

# Non-citable section tags (sections 33-42: ЛД, ПГО, ГВ, СМ/ФВ/FW, ОПГ, БВ, ВЗФ)
# References with these prefixes or suffixes must NEVER be turned into bare section-30 numbers.
NON_CITABLE_SECTION_TAG = (
    r"(?:ЛД|лд|LD|ld|ОПЛЗ|оплз|ОПЛиЗ|оплиз|ПГО|пго|PGO|pgo|ГВ|гв|GV|gv|"
    r"СМ|см|ФВ|фв|FW|fw|FV|fv|ОПГ|опг|OPG|opg|БВ|бв|BV|bv|ВЗФ|взф|VZF|vzf|"
    r"правил\w*\s+(?:гос|лид|опг|сем|войн|битв))"
)

# Regex for extracting rule references mentioned by players/admins:
# Captures optional non-citable prefixes/suffixes so we can explicitly reject them.
RULE_REF_PATTERN = re.compile(
    r"(?:(?P<other_prefix>" + NON_CITABLE_SECTION_TAG + r")(?:\s*[/\\]\s*зам\w*)?\s*|"
    r"(?P<prefix>ЖБ|жб|GZ|gz|ГЗ|гз)\s*)?"
    r"(?P<num>\d+(?:\.\d+)+)"
    r"(?:\s*\.?\s*(?:(?P<other_suffix>" + NON_CITABLE_SECTION_TAG + r")(?:\b|[/\\])|"
    r"(?P<suffix>ОП|оп|OP|op|GZ|gz|ГЗ|гз|ЖБ|жб|ППЖ|ппж)\b))?",
    re.UNICODE,
)

STANDALONE_GZ_PATTERN = re.compile(
    r"\b(?:GZ|ГЗ|зз|zz|зелён\w+\s+зон\w+|зелен\w+\s+зон\w+)\b",
    re.IGNORECASE,
)


def fix_cp1251_kazakh(text: str) -> str:
    """
    Fix cp1251 mojibake of Kazakh letters (ќ→қ, ґ→ғ, ѕ→ң, є→ә, ј→ө, ў→ұ, ї→ү).
    Preserves Ukrainian text (e.g. 'деємшиков', 'перевернітє', 'немає') when no Kazakh
    mojibake markers (ќ, ѕ, ј, ў, ѓ) or native Kazakh letters are present.
    """
    if not text:
        return ""
    has_strict_mojibake = any(ch in CP1251_STRICT_KAZAKH_CHARS for ch in text)
    has_native_kk = any(ch in KAZAKH_NATIVE_CHARS for ch in text)
    if not has_strict_mojibake and not has_native_kk and UKRAINIAN_MARKER_REGEX.search(text):
        return text
    return text.translate(CP1251_KAZAKH_MAP)


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize ticket text BEFORE language detection and RAG:
    1. Unicode NFC normalization
    2. Fix cp1251 Kazakh mojibake
    3. Clean up excessive whitespace and control characters
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text))
    norm = fix_cp1251_kazakh(norm)
    norm = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", norm)
    norm = re.sub(r"[ \t\u00a0]+", " ", norm)
    norm = re.sub(r"\n{3,}", "\n\n", norm)
    return norm.strip()


def _canonicalize_prefix(tag: Optional[str]) -> str:
    if not tag:
        return ""
    up = tag.strip().upper()
    if up in ("ОП", "OP"):
        return ""
    if up in ("GZ", "ГЗ"):
        return "GZ "
    if up in ("ЖБ", "ППЖ"):
        return "ЖБ "
    return ""


def extract_rule_refs(text: Optional[str], rule_refs_raw: Optional[Any] = None) -> List[str]:
    """
    Extract and normalize rule references from player text and rule_refs_raw
    into canonical format from sections 30-32: '2.1', '5.2', 'ЖБ 1.1.2', 'ЖБ 1.2.7', 'GZ 1.2', 'GZ'.
    References with suffixes/prefixes of sections 33-42 (ПГО, ГВ, СМ, ОПГ, БВ, ВЗФ, ЛД, ОПЛЗ, ФВ, FW)
    are explicitly ignored and NEVER converted into bare section-30 rule numbers.
    """
    combined_parts: List[str] = []
    if rule_refs_raw is not None:
        if isinstance(rule_refs_raw, list):
            combined_parts.extend(str(x) for x in rule_refs_raw if x is not None)
        else:
            combined_parts.append(str(rule_refs_raw))
    if text:
        combined_parts.append(text)

    combined = " ; ".join(combined_parts)
    if not combined.strip():
        return []

    seen = set()
    result: List[str] = []
    for match in RULE_REF_PATTERN.finditer(combined):
        if match.group("other_prefix") or match.group("other_suffix"):
            continue

        prefix = match.group("prefix")
        suffix = match.group("suffix")
        num = match.group("num")

        parts = num.split(".")
        if len(parts) > 3:
            continue
        if any(len(p) > 2 for p in parts):
            continue
        first_int = int(parts[0])
        if first_int == 0 or first_int > 5:
            continue

        # Ignore float numbers or times like 19.31 or 30.00
        if len(parts) == 2 and int(parts[1]) > 9:
            continue

        canon_pref = _canonicalize_prefix(prefix) or _canonicalize_prefix(suffix)
        # 3-part numbers starting with 1.1.x..1.5.x belong to section 31 (Правила жалоб -> ЖБ)
        if not canon_pref and len(parts) == 3 and parts[0] == "1" and parts[1] in ("1", "2", "3", "4", "5"):
            canon_pref = "ЖБ "

        canon = f"{canon_pref}{num}"
        if canon not in seen:
            seen.add(canon)
            result.append(canon)

    if rule_refs_raw and STANDALONE_GZ_PATTERN.search(str(rule_refs_raw)):
        if not any(r.startswith("GZ") for r in result):
            result.append("GZ")

    return result
