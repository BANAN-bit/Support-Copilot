import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.dedup import is_spam_or_flood_text
from app.langdetect import detect_language, resolve_reply_lang
from app.normalize import extract_rule_refs, normalize_text
from app.rules_index import RuleItem
from app.schemas import DraftReply, ExtractedData, TriageResponse

NICKNAME_REGEX = re.compile(r"\b([A-Z][a-zA-Z0-9]{1,20}_[A-Z][a-zA-Z0-9_]{1,25})\b")
CYRILLIC_CHECK_REGEX = re.compile(r"[а-яА-ЯёЁқғңәөұүіһҚҒҢӘӨҰҮІҺ]")

DM_ABBR_REGEX = re.compile(
    r"\b(?:дм|дб|ск|тк|рк|пг|мг|зз|гз|dm|db|sk|tk|rk|pg|mg|zz|gz|spawnkill|teamkill|deathmatch|driveby)\b",
    re.IGNORECASE,
)
CHEAT_WORD_REGEX = re.compile(
    r"\b(?:чит(?:ы|ер\w*|ами|ов|ить|ерить)?|спидхак\w*|аим\w*|автокликер\w*|макрос\w*|багоюз\w*|багаюз\w*|софт\w*|speedhack\w*|aimbot\w*|cheat\w*|soft|bagayuz)\b",
    re.IGNORECASE,
)
INSULT_WORD_REGEX = re.compile(
    r"(?:\bоск\w*|\bosk\w*|оскорб\w*|обзыва\w*|матер\w*|родн\w*|семь[юи]|токсич\w*|немощь|қорла\w*|балағат\w*|боқтық|haqorat|haqrat|oilamni|onadan|onasidan|sokdi|sokyapti|so[ʻ'‘’`]?k\w*|sokindi|sukindi|sokin\w*|insult\w*)",
    re.IGNORECASE,
)
# Report texts with a player ID number, violation abbreviation (дм, дб, нрп, чит, оп, ск, рк, тк),
# or rule reference must be classified as report.complaint and escalated (needs_human = True).
REPORT_VIOLATION_ABBR_REGEX = re.compile(
    r"\b(?:дм|дб|нрп|нонрп|чит\w*|софт\w*|оп|ск|рк|тк|оск\w*|пг|мг|зз|гз|"
    r"dm|db|nrp|nonrp|cheat\w*|soft\w*|op|sk|rk|tk|osk\w*|pg|mg|gz|zz)\b",
    re.IGNORECASE,
)
REPORT_PLAYER_ID_REGEX = re.compile(r"\b\d{1,4}(?:\s*ид|\s*id|\s*айді)?\b", re.IGNORECASE)
REPORT_ID_DM_REGEX = REPORT_VIOLATION_ABBR_REGEX

GZ_TEXT_REGEX = re.compile(
    r"\b(?:gz|гз|зз|zz|зелён\w*\s+зон\w*|зелен\w*\s+зон\w*|yashil\s+zona\w*|жасыл\s+аймақ\w*|green\s+zone)\b",
    re.IGNORECASE,
)

# Sub-topic regexes for deterministic draft differentiation within categories
FAMILY_INSULT_REGEX = re.compile(
    r"(?:родн\w*|семь[юия]|мам[уеы]|пап[уеы]|отц\w*|мать|отбас\w*|анам\w*|әкем\w*|шеше\w*|oila\w*|onam\w*|onadan|onasidan)",
    re.IGNORECASE,
)
NONRP_CUFFS_DM_REGEX = re.compile(
    r"(?:связал\w*|связк\w*|верёвк\w*|веревк\w*|наручник\w*|посадил\w*|повязал\w*|завизал\w*|нрп\s+коп|байлап|кісен|kishan\w*|qamoq\w*|bog['ʻ’`]?lab)",
    re.IGNORECASE,
)
DB_VEHICLE_DM_REGEX = re.compile(
    r"(?:\bдб\b|\bdb\b|driveby|задавил\w*|таран\w*|сбил\w+\s+машин\w*|көлікпен\s+бас\w*|mashinada\s+bos\w*)",
    re.IGNORECASE,
)
SK_TK_DM_REGEX = re.compile(
    r"(?:\bск\b|\bтк\b|\bsk\b|\btk\b|spawnkill|teamkill|на\s+спавне|спавнда|spavnda)",
    re.IGNORECASE,
)
BUG_ABUSE_CHEAT_REGEX = re.compile(
    r"(?:багоюз\w*|багаюз\w*|\bбаг\b|\bbag\b|bagayuz|текстур\w*\s+баг)",
    re.IGNORECASE,
)
AIM_MACRO_CHEAT_REGEX = re.compile(
    r"(?:аим\w*|aim\w*|автокликер\w*|макрос\w*|через\s+стен\w*)",
    re.IGNORECASE,
)
EXTORTION_SCAM_REGEX = re.compile(
    r"(?:вымога\w*|шантаж\w*|бопса\w*|tovlama\w*)",
    re.IGNORECASE,
)
ADS_RMT_REGEX = re.compile(
    r"(?:реклам\w*|пиар\w*|промокод\w*|жарнама\w*|reklama\w*)",
    re.IGNORECASE,
)
ADMIN_MUTE_REGEX = re.compile(
    r"(?:\bмут\w*|\bmute\w*|замутил\w*|дыбыс\w*\s+өшір\w*|chatimni\s+blok\w*)",
    re.IGNORECASE,
)
ADMIN_BAN_REGEX = re.compile(
    r"(?:\bбан\w*|\bban\w*|забанил\w*|разбан\w*|бұғат\w*|blokla\w*)",
    re.IGNORECASE,
)

# Guard regex: any mention of hack, donate, money, ban, or threat in ru, kk, uz, en, or translit
# must force needs_human = True and prevent auto-closing.
SENSITIVE_ESCALATION_REGEX = re.compile(
    r"(?:\bвзлом\w*|\bдонат\w*|\bденьг\w*|\bденег\b|\bбан\w*|\bзабан\w*|\bразбан\w*|\bугроз\w*|\bугрожа\w*|"
    r"\bбұз\w*|\bтөлем\w*|\bақша\w*|\bқорқыт\w*|"
    r"\bbuz\w*|\bdonat\w*|\bpul\w*|\bban\w*|\btahdid\w*|"
    r"\bvzlom\w*|\bdengi\b|\bdeneg\b|\bugroz\w*|\bhack\w*|\bmoney\b)",
    re.IGNORECASE,
)

# Patterns for the 4 substantive KB/rules report topics eligible for needs_human=False:
KB_VEHICLE_HELP_REGEX = re.compile(
    r"(?:переверн\w*|вытащ\w+\s+(?:пж\s+)?машин\w*|дастаньте\s+тачку|почин\w+\s+(?:мне\s+)?(?:тачку|машину|пожалуйста)|"
    r"улетел\s+в\s+речку|упал\s+в\s+воду.*машин\w*|эвакуатор\w*)",
    re.IGNORECASE,
)
KB_TELEPORT_HELP_REGEX = re.compile(
    r"(?:телепортир\w+\s+меня|\bтп\s+на\b|\bтп\s+ко\s+мне\b)",
    re.IGNORECASE,
)
# Strictly triggers ONLY on words about getting stuck, textures, or spawn (not 'я потерялся', 'не могу встать', etc.)
KB_STUCK_HELP_REGEX = re.compile(
    r"(?:застрял\w*|провалил\w*|текстур\w*|заспавн\w*|\bспавн\b|рестп\w*|"
    r"тұрып\s+қалдым|tekstur\w*|tushib\s+qoldi|stuck|debaixo\s+do\s+mapa|zastryal\w*|zaspavn\w*)",
    re.IGNORECASE,
)
# General non-stuck help requests ('я потерялся', 'поможітя я потірявся', 'не могу встать', 'в каком-то измерение')
# are classified as report.help, receive the neutral response, and are escalated to a human (needs_human = True).
GENERAL_HELP_REPORT_REGEX = re.compile(
    r"(?:потерял\w*|потіряв\w*|не\s+могу\s+встать|каком-то\s+измерени\w*)",
    re.IGNORECASE,
)
KB_NICK_QUESTION_REGEX = re.compile(
    r"(?:поменять\s+ник|сменить\s+ник|смена\s+нон\s*рп\s+ника|смена\s+ника|pomenyat\s+nik|nik\s+o['ʻ’`]?zgartirish|никнейм\s+өзгерту|change\s+nickname)",
    re.IGNORECASE,
)
KB_JB_QUESTION_REGEX = re.compile(
    r"(?:как\s+(?:мне\s+)?залить\s+жб|как\s+подать\s+жб|как\s+подать\s+жалобу|куда\s+писать\s+жалобу|shikoyat\s+yozish)",
    re.IGNORECASE,
)

SERVER_PATTERNS = [
    re.compile(r"(?:на\s+|#|\b)(\d{1,2})\s*(?:-м|-й|-ом)?\s*сервер[еау]?\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*servere\b", re.IGNORECASE),
    re.compile(r"\bсервер[еау]?\s*(?:№|#)?\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bserver\s*(?:№|#)?\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*(?:-ші|-шы)?\s*сервер(?:де|ге|ден)?\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*(?:-)?\s*server(?:da|ga|dan)?\b", re.IGNORECASE),
    re.compile(r"\bservidor\s*(?:№|#)?\s*(\d{1,2})\b", re.IGNORECASE),
]

URL_TOKEN_REGEX = re.compile(r"(?:<URL>|https?://\S+|youtu\.be/\S+|imgur\.com/\S+|yapx\.ru/\S+)", re.IGNORECASE)

THREAT_KEYWORDS = (
    "сватинг", "доксинг", "деанон", "убью в реале", "найду в жизни", "угрожает жизни", "угроза жизни",
    "swatting", "doxing", "real life threat",
)

MASS_BUG_KEYWORDS = (
    "массовый баг", "сервер упал", "дюп виртов", "дюп денег", "у всех пропали",
)

HACK_KEYWORDS = (
    "взлом", "взломали", "украли аккаунт", "поменяли пароль", "не могу войти",
    "пропала машина", "пропало имущество", "отвязали", "бұзып кетті", "бұзды",
    "кіре алмай", "мүлкім жоғалды", "құпиясөзімді өзгерт", "аккаунтымды ұрлап",
    "akkauntimni buz", "parolni almashtir", "kira olmayapman", "parolim ishlamayapti",
    "mulkim yo'qoldi", "akkauntim o'g'irlandi", "hacked", "cannot log in",
    "stolen account",
)

PAYMENT_KEYWORDS = (
    "донат", "задонатил", "платеж", "платёж", "оплата", "оплатил гк", "списались деньги",
    "не пришли", "не пришёл", "не пришел", "каспий", "сбп", "карта", "чеки",
    "төлем", "ақша түспеді", "теңге", "донат түспеді", "ақша шешілді",
    "donat", "pul yechib", "hisobimga tushmadi", "to'lov", "tolov qildim",
    "payment", "bought", "coins", "charged", "did not arrive",
    "pago", "tarjeta", "monedas", "pagamento", "cartão",
)

BAN_APPEAL_KEYWORDS = (
    "апелляция", "апел", "не согласен", "не согл", "пересмотреть наказание", "требую пересмотра",
    "пересмотреть жалобу", "снизить наказание", "ошибочный бан", "просто так бан", "бан просто так",
    "снимите тогда бан", "снимите бан", "разбаньте", "разбанье", "забанили просто так",
    "жалоба на администратора", "жалоба на админа", "админ выдал", "администратор выдал",
    "почему админ дает мут", "дала варн", "дали ванн", "неверно выдан варн", "админские нарушение",
    "опру", "не понял за что", "посадил без причини", "посадил когда я", "я даже не в кого не стрелял",
    "я его даже не убил", "где я текашил", "ну и кого же я убил", "нету софта у меня",
    "где софт у меня", "где вы видите", "где доказательство что я", "мут незачто",
    "обжаловать своё снятие", "блат, ангажированность", "покрывательство косяков",
    "non nrp adm", "nrp adm", "uzidan uzi blokladi", "besabab blokladi chatimni",
    "әкімші", "қате бан", "жазамды қайта қарау", "не үшін бан", "банмен келіспеймін", "әділетсіз бан",
    "nega mute beradi", "ban berdi", "warn beryabdi", "nohaq ban", "admin ustidan shikoyat", "jazoni qayta ko'rib",
    "admin ban", "unban appeal",
)

BARE_OFFTOPIC_REPORTS = {
    "помогите", "помогите пожалуйста", "помогите администрация", "помоги пожалуйста",
    "памагіте пжпжпжпж", "yordam", "yordam admin", "yordam bervorila", "+", "ау", "привет",
}

APPEAL_VERDICTS = {"appeal_approved", "appeal_rejected"}

TAXONOMY_EMPTY_RULE_CATEGORIES = {
    "complaint.other",
    "report.question",
    "report.complaint",
    "report.payment",
}

# Strictly aligned with data/taxonomy.json rules per category
PREFERRED_RULES_BY_CATEGORY: Dict[str, List[str]] = {
    "complaint.insult": ["2.1", "2.4", "2.6"],
    "complaint.cheat": ["5.2"],
    "complaint.dm": ["5.3", "5.4", "GZ"],
    "complaint.scam": ["4.1", "4.2"],
    "complaint.rmt_ads": ["4.4", "2.3"],
    "complaint.admin": ["3.4"],
    "complaint.invalid": ["ЖБ 1.1.2", "ЖБ 1.2.7"],
    "report.help": ["3.1"],
    "report.account": ["1.4"],
    "report.offtopic": ["3.1"],
}


def check_has_evidence(
    attachments: Optional[int],
    images_count: Optional[int],
    text: str,
) -> bool:
    att = max(0, int(attachments or 0))
    img = max(0, int(images_count or 0))
    if att > 0 or img > 0:
        return True
    if URL_TOKEN_REGEX.search(text or ""):
        return True
    return False


def extract_server_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in SERVER_PATTERNS:
        m = pat.search(text)
        if m:
            srv_num = m.group(1)
            return f"server_{srv_num}" if not srv_num.isdigit() else srv_num
    return None


def extract_accused_nickname(text: str, reported_nick: Optional[str] = None) -> Optional[str]:
    if reported_nick and isinstance(reported_nick, str):
        clean_rep = reported_nick.strip()
        if clean_rep and clean_rep.lower() not in ("null", "none", "-", "нет", "unknown_player"):
            return clean_rep
    if not text:
        return None
    matches = NICKNAME_REGEX.findall(text)
    if matches:
        return matches[0]
    return None


def has_substantive_kb_answer(category: str, text: str) -> Optional[str]:
    """
    Return the substantive KB/rules answer topic if the ticket is a report
    that has a direct, rule/KB-verified answer (stuck in textures/spawn, flip/recover vehicle,
    teleport rule 3.1, nickname rule 1.6, or how to file a complaint ЖБ 1.1.1).
    Returns None for everything else (including 'я потерялся', 'не могу встать', etc.).
    """
    if not text or category not in ("report.help", "report.question"):
        return None
    if category == "report.help":
        if KB_VEHICLE_HELP_REGEX.search(text):
            return "flip_vehicle"
        if KB_TELEPORT_HELP_REGEX.search(text):
            return "teleport_refusal"
        if KB_STUCK_HELP_REGEX.search(text):
            return "stuck_textures"
        return None
    if category == "report.question":
        if KB_NICK_QUESTION_REGEX.search(text):
            return "change_nick"
        if KB_JB_QUESTION_REGEX.search(text):
            return "how_to_complain"
        if GZ_TEXT_REGEX.search(text):
            return "green_zone_info"
        return None
    return None


def infer_category(
    text: str,
    kind: str,
    rule_refs: List[str],
    has_evidence: bool,
    is_spam: bool = False,
    verdict: Optional[str] = None,
    rule_refs_raw: Optional[Any] = None,
) -> str:
    """Deterministic category classifier aligned with kind ('complaint.*' vs 'report.*')."""
    raw_ref_str = str(rule_refs_raw or "") if rule_refs_raw is not None else ""
    combined = f"{text or ''} {raw_ref_str}".strip()
    low = combined.lower()
    text_low = (text or "").strip().lower()
    ref_set = set(rule_refs)
    verdict_low = (verdict or "").strip().lower()

    if kind == "complaint":
        if verdict_low in APPEAL_VERDICTS or "3.4" in ref_set or any(k in low for k in BAN_APPEAL_KEYWORDS):
            return "complaint.admin"
        if "ЖБ 1.1.2" in ref_set or "ЖБ 1.2.7" in ref_set or "проверяю форму" in low or "лвл не поднимается" in low:
            return "complaint.invalid"
        if "5.2" in ref_set or bool(CHEAT_WORD_REGEX.search(low)) or "баг qlb" in low or "bag qlb" in low or "ұшып жүр" in low:
            return "complaint.cheat"
        if ("2.3" in ref_set or "4.4" in ref_set) or any(
            k in low for k in ("продаёт вирты", "продает вирты", "за реал", "реальные деньги", "рекламирует", "магазин в чате", "rmt")
        ):
            return "complaint.rmt_ads"
        if ("4.1" in ref_set or "4.2" in ref_set) or any(
            k in low for k in ("обманул", "обман", "скам", "кинул на", "при обмене", "не отдал", "вымогает", "алдап", "алаяқтық", "aldab", "abmen", "firibgarlik", "scam", "obmanul")
        ):
            return "complaint.scam"
        if ("2.1" in ref_set or "2.4" in ref_set or "2.6" in ref_set) or bool(INSULT_WORD_REGEX.search(low)):
            return "complaint.insult"
        if (
            any(r in ref_set for r in ("5.3", "5.4", "GZ", "GZ 1.1", "GZ 1.2"))
            or bool(DM_ABBR_REGEX.search(low))
            or any(
                k in low
                for k in (
                    "убил", "стрелял", "стрельб", "зелёной зоне", "зеленой зоне", "задавил", "расстрел",
                    "атып өлтірді", "жасыл аймақ", "otib o'ldirdi", "oldirdi", "uldirdi", "uldrdi", "uryapdi",
                    "yashil zona", "ubil", "посадил без причин", "связал низа что", "наручники", "kishanlab", "qamoqga",
                )
            )
        ):
            return "complaint.dm"
        if detect_language(combined) == "und" or len(text_low) < 3 or verdict_low == "invalid_format":
            return "complaint.invalid"
        return "complaint.other"

    # kind == "report"
    if "1.4" in ref_set or any(k in low for k in HACK_KEYWORDS):
        return "report.account"
    if any(k in low for k in PAYMENT_KEYWORDS):
        return "report.payment"

    # Substantive help or question topics (change nick, how to complain, flip vehicle, stuck in textures)
    # plus general non-stuck help requests ('я потерялся', 'не могу встать', etc.)
    if KB_NICK_QUESTION_REGEX.search(text_low) or KB_JB_QUESTION_REGEX.search(text_low):
        return "report.question"
    if (
        KB_VEHICLE_HELP_REGEX.search(text_low)
        or KB_STUCK_HELP_REGEX.search(text_low)
        or KB_TELEPORT_HELP_REGEX.search(text_low)
        or GENERAL_HELP_REPORT_REGEX.search(text_low)
    ):
        return "report.help"

    clean_punct = re.sub(r"[!?.…,\s]+", " ", text_low).strip()
    if clean_punct in BARE_OFFTOPIC_REPORTS:
        return "report.offtopic"

    # Texts with a player number/ID, violation abbreviation (дм, дб, нрп, чит, оп, ск, рк, тк),
    # or rule reference are classified as report.complaint (and escalated to human).
    has_rule_ref_in_report = bool(ref_set) or bool(re.search(r"\b\d+\.\d+\b", text_low))
    has_violation_abbr = bool(REPORT_VIOLATION_ABBR_REGEX.search(text_low))
    has_player_id_num = bool(REPORT_PLAYER_ID_REGEX.search(text_low)) and "?" not in text_low

    if (
        has_rule_ref_in_report
        or has_violation_abbr
        or has_player_id_num
        or any(
            k in text_low
            for k in (
                "следите за", "расстреливает", "читерит", "оскорбляет", "тут дм", "меня дм",
                "связали", "зав'язали", "заломал", "у него что-то есть", "пристаёт", "пристаьот",
                "меня бьют", "убивают",
            )
        )
    ):
        return "report.complaint"

    if is_spam or is_spam_or_flood_text(text):
        return "report.offtopic"

    if "?" in text or any(k in text_low for k in ("как ", "что будет", "что такое", "где ", "когда ", "будет сегодня", "қалай", "деген не", "qanday", "nima", "how to", "what is", "kak ")):
        return "report.question"
    if len(clean_punct.split()) <= 2:
        return "report.offtopic"
    return "report.question"


def compute_priority(category: str, has_evidence: bool, text: str) -> str:
    low = (text or "").lower()
    if category in ("report.account", "report.payment"):
        return "p0"
    if any(k in low for k in THREAT_KEYWORDS) or any(k in low for k in MASS_BUG_KEYWORDS):
        return "p0"
    if category in ("complaint.cheat", "complaint.scam", "complaint.rmt_ads"):
        return "p1"
    if category in ("report.question", "report.offtopic", "complaint.invalid"):
        return "p3"
    if category in ("complaint.insult", "complaint.dm", "complaint.other"):
        return "p2" if has_evidence else "p3"
    if category in ("complaint.admin", "report.complaint", "report.help"):
        return "p2"
    return "p3"


def compute_needs_human(
    category: str,
    text: str,
    confidence: float,
    rag_has_answer: bool = True,
    verdict: Optional[str] = None,
) -> bool:
    """
    needs_human=False is permitted ONLY when:
    1) confidence >= 0.8 (при достаточной уверенности, §7.7),
    2) the draft reply contains a substantive answer confirmed by rules.json or the knowledge base
       (stuck in textures/spawn, flip vehicle, teleport rule 3.1, nickname change rule 1.6,
       how to submit a complaint ЖБ 1.1.1) and rag_has_answer is True, and
    3) the text contains no sensitive keywords (взлом, донат, деньги, бан, угроза).
    """
    if category.startswith("complaint."):
        return True
    if category in ("report.account", "report.payment", "report.complaint", "report.offtopic"):
        return True
    if (verdict or "").strip().lower() in APPEAL_VERDICTS:
        return True
    low = (text or "").lower()
    if SENSITIVE_ESCALATION_REGEX.search(low):
        return True
    if any(k in low for k in HACK_KEYWORDS) or any(k in low for k in PAYMENT_KEYWORDS):
        return True
    if any(k in low for k in BAN_APPEAL_KEYWORDS) or any(k in low for k in THREAT_KEYWORDS):
        return True
    if category in ("report.help", "report.question"):
        topic = has_substantive_kb_answer(category, text)
        if topic is not None and rag_has_answer and confidence >= 0.8:
            return False
    return True


def detect_question_kb_rule(text: str, rag_set: Set[str]) -> Optional[str]:
    """
    Check if a report.question text has a direct answer in rules.json / Knowledge Base:
    - Nickname rules -> 1.6
    - Green Zone rules -> GZ 1.1 or GZ
    - Filing a complaint -> ЖБ 1.1.1 or ЖБ 1.1.2
    """
    if not text:
        return None
    if KB_NICK_QUESTION_REGEX.search(text):
        if "1.6" in rag_set:
            return "1.6"
    if GZ_TEXT_REGEX.search(text):
        for gz_id in ("GZ 1.1", "GZ"):
            if gz_id in rag_set:
                return gz_id
    if KB_JB_QUESTION_REGEX.search(text):
        for jb_id in ("ЖБ 1.1.1", "ЖБ 1.1.2", "ЖБ 1.1"):
            if jb_id in rag_set:
                return jb_id
    return None


def select_best_citations(
    category: str,
    rule_refs: List[str],
    rag_items: List[RuleItem],
    has_evidence: bool,
    text: str = "",
    verdict: Optional[str] = None,
) -> List[str]:
    """
    Citation Selection:
    - For complaint.invalid and verdict=='invalid_format': cite ONLY ЖБ rules from taxonomy.json (or explicitly specified ЖБ rules).
    - For complaint.other: cite player's rule_refs ONLY if they exist in rules.json (and never add 3.1 unless explicitly specified by player).
    - For all categories with taxonomy rules (PREFERRED_RULES_BY_CATEGORY): cite ONLY rules belonging
      to that category's taxonomy list (or explicitly specified GZ sub-rules in complaint.dm).
    """
    rag_ids = [it.id for it in rag_items if it is not None]
    rag_set = set(rag_ids)
    if not rag_set:
        return []

    verdict_low = (verdict or "").strip().lower()
    valid_player_refs: List[str] = []
    for ref in rule_refs:
        if ref in rag_set and ref not in valid_player_refs:
            valid_player_refs.append(ref)

    # 1. For complaint.invalid or verdict=='invalid_format': cite ONLY ЖБ rules
    if category == "complaint.invalid" or verdict_low == "invalid_format":
        jb_player = [r for r in valid_player_refs if r.startswith("ЖБ ")]
        if jb_player:
            return jb_player[:2]
        if INSULT_WORD_REGEX.search(text or "") and "ЖБ 1.2.7" in rag_set:
            return ["ЖБ 1.2.7"]
        for jb_cand in ("ЖБ 1.1.2", "ЖБ 1.2.7"):
            if jb_cand in rag_set:
                return [jb_cand]
        any_jb = [r for r in rag_ids if r in ("ЖБ 1.1.2", "ЖБ 1.2.7")]
        return [any_jb[0]] if any_jb else []

    # 2. For categories with rules=[] in taxonomy.json (complaint.other, report.question, report.complaint, report.payment)
    if category in TAXONOMY_EMPTY_RULE_CATEGORIES:
        if category == "report.question":
            kb_rule = detect_question_kb_rule(text, rag_set)
            if kb_rule:
                return [kb_rule]
            return [r for r in valid_player_refs if r != "3.1"][:1]
        if category == "complaint.other":
            return [r for r in valid_player_refs if r != "3.1"][:2]
        if category == "report.complaint":
            return [r for r in valid_player_refs if r != "3.1"][:1]
        return []

    # 3. For categories with taxonomy rules: cite ONLY rules belonging to that category in taxonomy.json
    # (or explicitly specified GZ sub-rules in valid_player_refs for complaint.dm)
    cat_prefs = list(PREFERRED_RULES_BY_CATEGORY.get(category, []))
    if category == "complaint.dm":
        explicit_gz = [r for r in valid_player_refs if r.startswith("GZ")]
        if explicit_gz:
            return explicit_gz[:2]
        if GZ_TEXT_REGEX.search(text or ""):
            cat_prefs = ["GZ", "5.3", "5.4"]

    matching_cat_refs = [r for r in valid_player_refs if r in cat_prefs]
    if matching_cat_refs:
        return matching_cat_refs[:2]

    primary_cat_rule = next((c for c in cat_prefs if c in rag_set), None)
    return [primary_cat_rule] if primary_cat_rule else []


def select_best_citation(
    category: str,
    rule_refs: List[str],
    rag_items: List[RuleItem],
    has_evidence: bool,
    text: str = "",
    verdict: Optional[str] = None,
) -> Optional[str]:
    cites = select_best_citations(category, rule_refs, rag_items, has_evidence, text=text, verdict=verdict)
    return cites[0] if cites else None


def calibrate_confidence(
    text: str,
    lang: str,
    category: str,
    rule_refs: List[str],
    cited_id: Optional[str],
    has_evidence: bool,
    is_spam_or_dup: bool,
) -> float:
    """
    Heuristic confidence score (эвристическая оценка уверенности) in [0.68, 0.93].
    Does NOT depend on text length (word_count); auto-closing requires confidence >= 0.8 (§7.7).
    """
    if lang == "und":
        return 0.68
    if is_spam_or_dup:
        return 0.75

    score = 0.80
    if rule_refs:
        score += 0.06
    if cited_id:
        score += 0.04
    if has_evidence:
        score += 0.03
    if has_substantive_kb_answer(category, text) is not None:
        score += 0.02
    if lang == "lat":
        score -= 0.04

    return round(max(0.68, min(0.93, score)), 2)


def _format_cited_label(cited_ids: List[str], lang: str) -> str:
    if not cited_ids:
        return ""
    joined = ", ".join(cited_ids)
    if lang == "ru":
        return f"пункт {joined}" if len(cited_ids) == 1 else f"пункты {joined}"
    if lang == "kk":
        return f"{joined} тармағы" if len(cited_ids) == 1 else f"{joined} тармақтары"
    if lang == "uz":
        return f"{joined} qoidasi" if len(cited_ids) == 1 else f"{joined} qoidalari"
    if lang == "es":
        return f"regla {joined}" if len(cited_ids) == 1 else f"reglas {joined}"
    if lang == "pt":
        return f"regra {joined}" if len(cited_ids) == 1 else f"regras {joined}"
    return f"rule {joined}" if len(cited_ids) == 1 else f"rules {joined}"


def _format_cited_label_kk_accusative(cited_ids: List[str]) -> str:
    """Grammatically correct Kazakh accusative form ('тармағын' / 'тармақтарын') before 'көрсеттіңіз'."""
    if not cited_ids:
        return ""
    joined = ", ".join(cited_ids)
    return f"{joined} тармағын" if len(cited_ids) == 1 else f"{joined} тармақтарын"


def _format_cited_label_uz_accusative(cited_ids: List[str]) -> str:
    """Grammatically correct Uzbek accusative form ('qoidasini' / 'qoidalarini') before 'ko'rsatdingiz'."""
    if not cited_ids:
        return ""
    joined = ", ".join(cited_ids)
    return f"{joined} qoidasini" if len(cited_ids) == 1 else f"{joined} qoidalarini"


def _build_rule_clause(
    cited_list: List[str],
    player_rule_refs: Optional[List[str]],
    lbl_ru: str,
    lbl_kk: str,
    lbl_kk_acc: str,
    lbl_uz: str,
    lbl_uz_acc: str,
    lbl_en: str,
    lbl_es: str,
    lbl_pt: str,
) -> Dict[str, str]:
    """
    If the player explicitly specified the cited rule(s) in rule_refs, use 'вы указали пункт X правил'
    (and equivalents in kk/uz). Otherwise, reference the category rule neutrally in parentheses.
    """
    if not cited_list:
        return {"ru": "", "kk": "", "uz": "", "en": "", "es": "", "pt": ""}
    player_set = set(player_rule_refs or [])
    if player_set and all(c in player_set for c in cited_list):
        return {
            "ru": f" (вы указали {lbl_ru} правил)",
            "kk": f" (сіз ереженің {lbl_kk_acc} көрсеттіңіз)",
            "uz": f" (siz {lbl_uz_acc} ko'rsatdingiz)",
            "en": f" (you specified {lbl_en})",
            "es": f" ({lbl_es})",
            "pt": f" ({lbl_pt})",
        }
    return {
        "ru": f" ({lbl_ru})",
        "kk": f" ({lbl_kk})",
        "uz": f" ({lbl_uz})",
        "en": f" ({lbl_en})",
        "es": f" ({lbl_es})",
        "pt": f" ({lbl_pt})",
    }


def _build_complaint_followup(
    nickname: Optional[str],
    has_evidence: bool,
    server: Optional[str],
    safe_nick_uz: str,
) -> Dict[str, str]:
    """
    Deterministic follow-up sentence(s) for complaint categories based on:
    - presence of accused nickname,
    - presence of evidence (screenshots/video),
    - presence of server number.
    """
    if has_evidence and nickname:
        if server:
            return {
                "ru": f" Приложенные материалы по серверу {server} переданы администратору на проверку.",
                "kk": f" {server}-сервер бойынша тіркелген материалдар әкімшіге тексеруге берілді.",
                "uz": f" {server}-server bo'yicha ilova qilingan materiallar ma'murga tekshirish uchun yuborildi.",
                "en": " Your attached materials have been forwarded to an administrator for review.",
                "es": " Sus materiales adjuntos han sido enviados a un administrador para su revisión.",
                "pt": " Seus materiais anexados foram encaminhados a um administrador para análise.",
            }
        return {
            "ru": " Приложенные материалы переданы администратору на проверку. Если сервер или время ситуации не указаны, вы можете дополнить обращение.",
            "kk": " Тіркелген материалдар әкімшіге тексеруге берілді. Егер сервер нөмірі немесе оқиға уақыты көрсетілмесе, өтінішті толықтыра аласыз.",
            "uz": " Ilova qilingan materiallar ma'murga tekshirish uchun yuborildi. Agar server raqami yoki holat vaqti ko'rsatilmagan bo'lsa, murojaatga qo'shimcha yozishingiz mumkin.",
            "en": " Your attached materials have been forwarded to an administrator for review.",
            "es": " Sus materiales adjuntos han sido enviados a un administrador para su revisión.",
            "pt": " Seus materiais anexados foram encaminhados a um administrador para análise.",
        }
    if has_evidence and not nickname:
        return {
            "ru": " Приложенные материалы переданы администратору на проверку. Пожалуйста, уточните игровой никнейм нарушителя, а также сервер и время ситуации.",
            "kk": " Тіркелген материалдар әкімшіге тексеруге берілді. Тәртіп бұзушының ойындағы никнеймін, сондай-ақ сервер мен оқиға уақытын нақтылап жазуыңызды сұраймыз.",
            "uz": " Ilova qilingan materiallar ma'murga tekshirish uchun yuborildi. Iltimos, qoidabuzarning o'yindagi nikini hamda server va holat vaqtini ko'rsating.",
            "en": " Your attached materials have been forwarded to an administrator for review.",
            "es": " Sus materiales adjuntos han sido enviados a un administrador para su revisión.",
            "pt": " Seus materiais anexados foram encaminhados a um administrador para análise.",
        }
    if not has_evidence and nickname:
        return {
            "ru": " Ваш запрос передан администратору. Для объективной проверки, пожалуйста, приложите скриншот или видеозапись ситуации и укажите сервер и время.",
            "kk": " Сұранысыңыз әкімшіге берілді. Жан-жақты тексеру үшін жағдайдың скриншотын немесе бейнежазбасын тіркеп, сервер мен уақытты көрсетуіңізді сұраймыз.",
            "uz": " So'rovingiz ma'murga yuborildi. Xolis tekshirish uchun holat aks etgan skrinshot yoki videoyozuvni ilova qilib, server va vaqtni ko'rsatishingizni so'raymiz.",
            "en": " Your request has been forwarded to an administrator — to review your complaint, please attach a screenshot or video recording of the incident.",
            "es": " Su solicitud ha sido enviada a un administrador; por favor adjunte una captura de pantalla o grabación de video.",
            "pt": " Seu pedido foi encaminhado a um administrador; por favor anexe uma captura de tela ou gravação de vídeo.",
        }
    return {
        "ru": " Ваш запрос передан администратору. Пожалуйста, укажите точный никнейм нарушителя, номер сервера и время ситуации, а также приложите скриншот или видеозапись.",
        "kk": " Сұранысыңыз әкімшіге берілді. Тәртіп бұзушының нақты никнеймін, сервер нөмірі мен оқиға уақытын көрсетіп, скриншот немесе бейнежазба тіркеуіңізді сұраймыз.",
        "uz": " So'rovingiz ma'murga yuborildi. Iltimos, qoidabuzarning aniq nikini, server raqami va holat vaqtini ko'rsating hamda skrinshot yoki videoyozuv ilova qiling.",
        "en": " Your request has been forwarded to an administrator — to review your complaint, please attach a screenshot or video recording of the incident.",
        "es": " Su solicitud ha sido enviada a un administrador; por favor adjunte una captura de pantalla o grabación de video.",
        "pt": " Seu pedido foi encaminhado a um administrador; por favor anexe uma captura de tela ou gravação de vídeo.",
    }


def build_safe_draft_text(
    category: str,
    reply_lang: str,
    nickname: Optional[str],
    cited_id: Optional[Any],
    has_evidence: bool,
    rag_items: Optional[List[RuleItem]] = None,
    verdict: Optional[str] = None,
    player_text: str = "",
    player_rule_refs: Optional[List[str]] = None,
    server: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Build category-specific, sub-scenario-aware, evidence-aware, and verdict-aware draft replies
    across ru, kk, uz, en, es, pt.
    - Differentiates deterministically by sub-topic, nickname presence, evidence presence, server presence,
      and whether the cited rule was explicitly specified by the player ('вы указали пункт X правил').
    - No promises of punishment, unban, refund, or timeframe.
    - Pure Latin script in uz drafts (except official Cyrillic rule IDs such as 'ЖБ 1.1.2' when cited).
    - Rule numbers in draft text come strictly from cited_list.
    """
    if isinstance(cited_id, list):
        cited_list = [str(c) for c in cited_id if c]
    elif isinstance(cited_id, str) and cited_id:
        cited_list = [cited_id]
    else:
        cited_list = []

    verdict_low = (verdict or "").strip().lower()
    p_text = player_text or ""
    if server is None:
        server = extract_server_from_text(p_text)

    safe_nick_uz = nickname if (nickname and not CYRILLIC_CHECK_REGEX.search(nickname)) else None

    nick_ru = f" на игрока {nickname}" if nickname else ""
    nick_kk = f" ({nickname} бойынша)" if nickname else ""
    nick_uz = f" ({safe_nick_uz} bo'yicha)" if safe_nick_uz else ""
    nick_en = f" regarding player {nickname}" if nickname else ""

    lbl_ru = _format_cited_label(cited_list, "ru")
    lbl_kk = _format_cited_label(cited_list, "kk")
    lbl_kk_acc = _format_cited_label_kk_accusative(cited_list)
    lbl_uz = _format_cited_label(cited_list, "uz")
    lbl_uz_acc = _format_cited_label_uz_accusative(cited_list)
    lbl_en = _format_cited_label(cited_list, "en")
    lbl_es = _format_cited_label(cited_list, "es")
    lbl_pt = _format_cited_label(cited_list, "pt")

    rc = _build_rule_clause(
        cited_list, player_rule_refs,
        lbl_ru, lbl_kk, lbl_kk_acc, lbl_uz, lbl_uz_acc, lbl_en, lbl_es, lbl_pt,
    )

    # 1. Invalid format verdict or complaint.invalid category (Sources: ЖБ 1.1.2, ЖБ 1.2.7)
    if verdict_low == "invalid_format" or category == "complaint.invalid":
        r_ru = f" ({lbl_ru})" if lbl_ru else ""
        r_kk = f" ({lbl_kk})" if lbl_kk else ""
        r_uz = f" ({lbl_uz})" if lbl_uz else ""
        r_en = f" ({lbl_en})" if lbl_en else ""
        r_es = f" ({lbl_es})" if lbl_es else ""
        r_pt = f" ({lbl_pt})" if lbl_pt else ""
        templates = {
            "ru": f"Здравствуйте! Ваша жалоба{nick_ru} не может быть принята к рассмотрению, так как нарушены правила подачи жалоб{r_ru}. Пожалуйста, оформите обращение в разделе жалоб лаунчера по требованиям: укажите корректный никнейм нарушителя, точное описание сути произошедшего без оскорблений и приложите полные необрезанные доказательства (скриншоты или видео).",
            "kk": f"Сәлеметсіз бе! Сіздің шағымыңыз{nick_kk} шағым беру ережелеріне{r_kk} сәйкес толтырылмағандықтан қарауға қабылданбайды. Өтінішті талаптарға сай қайта жолдауыңызды сұраймыз: тәртіп бұзушының нақты никнеймін көрсетіңіз, жағдайды балағат сөзсіз сипаттаңыз және толық дәлелдемелерді (скриншот немесе бейнежазба) тіркеңіз.",
            "uz": f"Assalomu alaykum! Shikoyatingiz{nick_uz} topshirish qoidalariga{r_uz} mos ravishda rasmiylashtirilmagani sababli ko'rib chiqilmaydi. Iltimos, murojaatni talablarga muvofiq qayta yuboring: qoidabuzarning aniq nikini ko'rsating, holatni haqoratlarsiz tushuntiring hamda to'liq dalillarni (skrinshot yoki videoyozuv) ilova qiling.",
            "en": f"Hello! Your complaint{nick_en} cannot be reviewed as it does not meet the complaint submission rules{r_en}. Please resubmit with the correct player nickname, a clear description without insults, and full unedited screenshots or video.",
            "es": f"¡Hola! Su queja no cumple con las reglas de presentación{r_es}. Por favor, envíela de nuevo indicando el nombre exacto del jugador, una descripción sin insultos y capturas completas o video.",
            "pt": f"Olá! Sua reclamação não atende às regras de envio{r_pt}. Por favor, envie novamente informando o nickname exato do jogador, uma descripción sem ofensas e capturas completas ou vídeo.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 2. Account hack / access / lost property (report.account)
    # Task 1: remind not to share password with third parties and ask to describe when the problem started
    if category == "report.account":
        r_ru = rc["ru"]
        r_kk = rc["kk"]
        r_uz = rc["uz"]
        r_en = f" ({lbl_en})" if lbl_en else ""
        r_es = f" ({lbl_es})" if lbl_es else ""
        r_pt = f" ({lbl_pt})" if lbl_pt else ""
        srv_ask_ru = f" на сервере {server}" if server else " (укажите также ваш никнейм и номер сервера)"
        srv_ask_kk = f" ({server}-сервер)" if server else " (сондай-ақ никнейміңіз бен сервер нөмірін көрсетіңіз)"
        srv_ask_uz = f" ({server}-server)" if server else " (shuningdek, nikingiz va server raqamini ko'rsating)"
        templates = {
            "ru": f"Здравствуйте! Ваш запрос по вопросу доступа к аккаунту или сохранности имущества{r_ru} передан специалисту технической поддержки. Напоминаем: никому не передавайте свой пароль и коды подтверждения третьим лицам. Пожалуйста, опишите, когда именно началась проблема{srv_ask_ru}.",
            "kk": f"Сәлеметсіз бе! Аккаунтқа кіру немесе мүлік қауіпсіздігі бойынша сұранысыңыз{r_kk} техникалық қолдау маманына берілді. Құпиясөзіңізді және растау кодтарын үшінші тұлғаларға ешқашан бермеуді еске саламыз. Мәселе қашан басталғанын сипаттап жазуыңызды сұраймыз{srv_ask_kk}.",
            "uz": f"Assalomu alaykum! Akkauntga kirish yoki mulk xavfsizligi bo'yicha so'rovingiz{r_uz} texnik yordam mutaxassisiga yuborildi. Parolingiz va tasdiqlash kodlarini uchinchi shaxslarga bermaslikni eslatib o'tamiz. Iltimos, muammo qachon boshlanganini batafsil yozib qoldiring{srv_ask_uz}.",
            "en": f"Hello! Your request regarding account access{r_en} has been forwarded to a technical support specialist. Please note that under project rules, players are personally responsible for account security.",
            "es": f"¡Hola! Su solicitud sobre el acceso a la cuenta{r_es} ha sido enviada a un especialista de soporte técnico.",
            "pt": f"Olá! Seu pedido sobre o acesso à conta{r_pt} foi encaminhado a um especialista de suporte técnico.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 3. Payment / donate (report.payment)
    # Task 1: ask for payment date and game nickname, without promising a refund
    if category == "report.payment":
        r_ru = rc["ru"]
        r_kk = rc["kk"]
        r_uz = rc["uz"]
        r_en = f" ({lbl_en})" if lbl_en else ""
        r_es = f" ({lbl_es})" if lbl_es else ""
        r_pt = f" ({lbl_pt})" if lbl_pt else ""
        if has_evidence:
            templates = {
                "ru": f"Здравствуйте! Ваш запрос по вопросу платежа или зачисления доната{r_ru} и приложенные материалы переданы специалисту технической поддержки. Пожалуйста, укажите точную дату платежа, ваш игровой никнейм и номер сервера для проверки транзакции.",
                "kk": f"Сәлеметсіз бе! Төлем немесе донат түсуі бойынша сұранысыңыз{r_kk} және тіркелген материалдар техникалық қолдау маманына берілді. Транзакцияны тексеру үшін төлем күнін, ойындағы никнейміңізді және сервер нөмірін көрсетуіңізді сұраймыз.",
                "uz": f"Assalomu alaykum! To'lov yoki donat tushishi bo'yicha so'rovingiz{r_uz} hamda ilova qilingan materiallar texnik yordam mutaxassisiga yuborildi. Tranzaksiyani tekshirish uchun to'lov sanasini, o'yindagi nikingizni va server raqamini ko'rsatishingizni so'raymiz.",
                "en": f"Hello! Your payment request{r_en} and the attached materials have been forwarded to a technical support specialist.",
                "es": f"¡Hola! Su solicitud sobre el pago{r_es} y los materiales adjuntos han sido enviados a un especialista de soporte técnico.",
                "pt": f"Olá! Seu pedido sobre o pagamento{r_pt} e os materiais anexados foram encaminhados a um especialista de suporte técnico.",
            }
        else:
            templates = {
                "ru": f"Здравствуйте! Ваш запрос по вопросу платежа или зачисления доната{r_ru} передан специалисту технической поддержки. Пожалуйста, укажите дату и время платежа, ваш игровой никнейм и сервер, а также приложите скриншот чека оплаты.",
                "kk": f"Сәлеметсіз бе! Төлем немесе донат түсуі бойынша сұранысыңыз{r_kk} техникалық қолдау маманына берілді. Төлем жасалған күнді, ойындағы никнейміңіз бен серверді көрсетіп, төлем түбіртегінің скриншотын тіркеуіңізді сұраймыз.",
                "uz": f"Assalomu alaykum! To'lov yoki donat tushishi bo'yicha so'rovingiz{r_uz} texnik yordam mutaxassisiga yuborildi. Iltimos, to'lov sanasini, o'yindagi nikingiz va serverni ko'rsating hamda to'lov cheki skrinshotini ilova qiling.",
                "en": f"Hello! Your payment request{r_en} has been forwarded to a technical support specialist. If you have a screenshot or video recording, please attach it to your inquiry.",
                "es": f"¡Hola! Su solicitud sobre el pago{r_es} ha sido enviada a un especialista de soporte técnico. Si tiene una captura o video, por favor adjúntelo.",
                "pt": f"Olá! Seu pedido sobre o pagamento{r_pt} foi encaminhado a um especialista de suporte técnico. Se tiver uma captura de tela ou vídeo, por favor anexe.",
            }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 4. Appeals and Admin complaints (complaint.admin)
    # Task 1: inform that the appeal is forwarded to senior administration, without evaluating punishment validity
    if category == "complaint.admin" or verdict_low in APPEAL_VERDICTS:
        r_ru = rc["ru"]
        r_kk = rc["kk"]
        r_uz = rc["uz"]
        r_en = f" ({lbl_en})" if lbl_en else ""
        r_es = f" ({lbl_es})" if lbl_es else ""
        r_pt = f" ({lbl_pt})" if lbl_pt else ""

        if ADMIN_MUTE_REGEX.search(p_text):
            sub_ru, sub_kk, sub_uz = "выданного мута или действий администратора", "берілген мутқа немесе әкімші әрекетіне", "berilgan mut yoki ma'mur harakati"
        elif ADMIN_BAN_REGEX.search(p_text):
            sub_ru, sub_kk, sub_uz = "блокировки аккаунта или действий администратора", "аккаунтты бұғаттауға немесе әкімші әрекетіне", "akkaunt bloklanishi yoki ma'mur harakati"
        else:
            sub_ru, sub_kk, sub_uz = "действий или наказания администратора", "әкімшінің әрекетіне немесе берілген жазаға", "ma'mur harakati yoki berilgan jazo"

        if has_evidence and nickname:
            fup_ru = " Приложенные материалы переданы на рассмотрение; если время выдачи наказания и сервер не указаны, вы можете дополнить обращение."
            fup_kk = " Тіркелген материалдар қарауға берілді; егер жаза берілген уақыт пен сервер көрсетілмесе, өтінішті толықтыра аласыз."
            fup_uz = " Ilova qilingan materiallar ko'rib chiqish uchun topshirildi; agar jazo vaqti va server ko'rsatilmagan bo'lsa, murojaatni to'ldirishingiz mumkin."
        elif has_evidence and not nickname:
            fup_ru = " Приложенные материалы переданы на рассмотрение. Пожалуйста, уточните никнейм администратора, номер сервера и время выдачи наказания."
            fup_kk = " Тіркелген материалдар қарауға берілді. Әкімшінің никнеймін, сервер нөмірін және жаза берілген уақытты нақтылап жазуыңызды сұраймыз."
            fup_uz = " Ilova qilingan materiallar ko'rib chiqish uchun topshirildi. Iltimos, ma'mur nikini, server raqamini va jazo berilgan vaqtni ko'rsating."
        elif not has_evidence and nickname:
            fup_ru = " Для рассмотрения обращения, пожалуйста, приложите скриншот окна наказания или видеозапись ситуации и укажите сервер."
            fup_kk = " Өтінішті қарау үшін жаза терезесінің скриншотын немесе жағдайдың бейнежазбасын тіркеп, серверді көрсетуіңізді сұраймыз."
            fup_uz = " Murojaatni ko'rib chiqish uchun jazo oynasi skrinshotini yoki holat videoyozuvini ilova qilib, serverni ko'rsatishingizni so'raymiz."
        else:
            fup_ru = " Пожалуйста, укажите никнейм администратора, сервер и время выдачи наказания, а также приложите скриншот окна наказания или видео."
            fup_kk = " Әкімшінің никнеймін, сервер мен жаза уақытын көрсетіп, жаза терезесінің скриншотын немесе бейнежазба тіркеуіңізді сұраймыз."
            fup_uz = " Iltimos, ma'mur nikini, server va jazo vaqtini ko'rsating hamda jazo oynasi skrinshotini yoki videoni ilova qiling."

        templates = {
            "ru": f"Здравствуйте! Ваше обращение по обжалованию {sub_ru}{nick_ru}{r_ru} передано старшей администрации.{fup_ru}",
            "kk": f"Сәлеметсіз бе! Сіздің {sub_kk} қатысты өтінішіңіз{nick_kk}{r_kk} аға әкімшілікке берілді.{fup_kk}",
            "uz": f"Assalomu alaykum! Sizning {sub_uz} yuzasidan murojaatingiz{nick_uz}{r_uz} katta ma'muriyatga yuborildi.{fup_uz}",
            "en": f"Hello! Your appeal regarding administrator actions{nick_en}{r_en} and the attached materials have been forwarded to a senior administration specialist." if has_evidence else f"Hello! Your appeal regarding administrator actions{nick_en}{r_en} has been forwarded to a senior administration specialist. To review your appeal, please attach a screenshot of the punishment or a video recording.",
            "es": f"¡Hola! Su apelación sobre la administración{r_es} ha sido enviada a un especialista de la administración superior.",
            "pt": f"Olá! Sua apelação sobre a administração{r_pt} foi encaminhada a um especialista da administração superior.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 5. report.help -> Sources:
    # - Vehicle tow button on speedometer / vehicle menu: tickets 4a6eb05bf6e93c72, 970940f5e5a8aafe, 5678552f794f526c, f96cbab7ff387d4d, 10c983a6d2c6597b, 1f71a442ce53cca4 + rule 3.1
    # - Teleport refusal: rule 3.1 + tickets 1846cf5299204c4c, 695b55dbaeff4345, 1e68266f0a8bbc99
    # - Stuck in textures: rule 3.1 + tickets 319b7349e324c677, 2a2a9134555456ba, f7f8feb05949e933, 4c2822545ebe517b
    if category == "report.help":
        topic = has_substantive_kb_answer(category, p_text)
        if cited_list and topic == "flip_vehicle":
            templates = {
                "ru": f"Здравствуйте! По вашему запросу о помощи с транспортом{rc['ru']}: вы можете самостоятельно перевернуть, починить или вернуть машину через кнопку эвакуатора на спидометре (в меню транспорта).",
                "kk": f"Сәлеметсіз бе! Көлік бойынша көмек сұрауыңызға{rc['kk']} сәйкес: көлікті спидометрдегі эвакуатор батырмасы (көлік мәзірі) арқылы өз бетіңізше аударып, жөндеп немесе қайтара аласыз.",
                "uz": f"Assalomu alaykum! Transport bo'yicha yordam so'rovingizga{rc['uz']} muvofiq: mashinani spidometrdagi evakuator tugmasi (transport menyusi) orqali mustaqil ravishda o'nglab yoki ta'mirlab olishingiz mumkin.",
                "en": f"Hello! Regarding your vehicle assistance request ({lbl_en}): you can flip, repair, or recover your vehicle yourself using the tow button on the speedometer.",
                "es": f"¡Hola! Según su solicitud ({lbl_es}), puede recuperar o reparar su vehículo usando el botón de grúa en el velocímetro.",
                "pt": f"Olá! Sobre seu pedido ({lbl_pt}), você pode recuperar ou reparar seu veículo usando o botão de guincho no velocímetro.",
            }
            return templates.get(reply_lang, templates["ru"]), cited_list
        if cited_list and topic == "teleport_refusal":
            templates = {
                "ru": f"Здравствуйте! Согласно правилам обращения в репорт{rc['ru']}, администрация не телепортирует игроков и транспорт по просьбе в репорт (просить телепорт, игровую валюту или имущество запрещено).",
                "kk": f"Сәлеметсіз бе! Репорт ережесіне{rc['kk']} сәйкес әкімшілік сұраныс бойынша телепортация жасамайды (репортта телепорт, ойын валютасын немесе мүлік сұрауға тыйым салынады).",
                "uz": f"Assalomu alaykum! Report qoidasiga{rc['uz']} muvofiq ma'muriyat so'rov bo'yicha teleport qilmaydi (reportda teleport, o'yin valyutasi yoki mulk so'rash taqiqlanadi).",
                "en": f"Hello! Under the report rules ({lbl_en}), administration does not provide teleportation on request (asking for teleports, currency, or items in report is prohibited).",
                "es": f"¡Hola! Según las reglas ({lbl_es}), la administración no realiza teletransportes a pedido.",
                "pt": f"Olá! Pelas regras ({lbl_pt}), a administração não realiza teletransportes a pedido.",
            }
            return templates.get(reply_lang, templates["ru"]), cited_list
        if cited_list and topic == "stuck_textures":
            templates = {
                "ru": f"Здравствуйте! По правилам обращения в репорт{rc['ru']} ваше сообщение о застрявшем персонаже принято — ваш запрос передан администратору.",
                "kk": f"Сәлеметсіз бе! Репорт ережесіне{rc['kk']} сәйкес кейіпкердің тұрып қалғаны туралы хабарламаңыз қабылданды — сұранысыңыз әкімшіге берілді.",
                "uz": f"Assalomu alaykum! Report qoidasiga{rc['uz']} muvofiq qahramon qolib ketgani haqidagi xabaringiz qabul qilindi — so'rovingiz ma'murga yuborildi.",
                "en": f"Hello! Under the report rules ({lbl_en}), your message regarding a stuck character has been received — your request has been forwarded to an administrator.",
                "es": f"¡Hola! Según las reglas ({lbl_es}), su mensaje sobre un personaje atascado ha sido recibido — su solicitud ha sido enviada a un administrador.",
                "pt": f"Olá! Pelas regras ({lbl_pt}), sua mensagem sobre um personagem preso foi recebida — seu pedido foi encaminhado a um administrador.",
            }
            return templates.get(reply_lang, templates["ru"]), cited_list
        r_en = f" ({lbl_en})" if lbl_en else ""
        r_es = f" ({lbl_es})" if lbl_es else ""
        r_pt = f" ({lbl_pt})" if lbl_pt else ""
        templates = {
            "ru": f"Здравствуйте! Ваш запрос передан администратору{rc['ru']}. Пожалуйста, уточните, что именно произошло и на каком сервере вы находитесь.",
            "kk": f"Сәлеметсіз бе! Ойындағы көмек бойынша сұранысыңыз{rc['kk']} әкімшіге берілді. Нақты не болғанын және қай серверде екеніңізді жазуыңызды сұраймыз.",
            "uz": f"Assalomu alaykum! O'yindagi yordam bo'yicha so'rovingiz{rc['uz']} ma'murga yuborildi. Iltimos, aynan nima bo'lganini va qaysi serverda ekanligingizni aniqlashtiring.",
            "en": f"Hello! Your request has been forwarded to an administrator{r_en}.",
            "es": f"¡Hola! Su solicitud ha sido enviada a un administrador{r_es}.",
            "pt": f"Olá! Seu pedido foi encaminhado a um administrador{r_pt}.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 6. report.question -> Sources:
    # - Rule 1.6 (prohibited nicknames similar to admins/media partners or with insults)
    # - Rule ЖБ 1.1.1 + tickets 18891139636a7e5f, bca8268464644e13 (complaints submitted via launcher with evidence)
    # - Rule GZ 1.1 / GZ 2.1-2.3 (Green Zone rules and locations)
    if category == "report.question":
        rule_obj = None
        if cited_list and rag_items:
            for it in rag_items:
                if it.id == cited_list[0]:
                    rule_obj = it
                    break
        if cited_list and rule_obj:
            cid0 = cited_list[0]
            if cid0 == "1.6":
                templates = {
                    "ru": f"Здравствуйте! По правилам проекта{rc['ru']}, запрещены никнеймы, схожие с существующими (включая администраторов и медиа-партнёров), а также никнеймы с оскорблениями или ненормативной лексикой.",
                    "kk": f"Сәлеметсіз бе! Жоба ережесіне{rc['kk']} сәйкес бұрыннан бар ойыншылардың (соның ішінде әкімшілер мен медиа-серіктестердің) атына ұқсас, сондай-ақ қорлау немесе балағат сөздері бар никнеймдерге тыйым салынады.",
                    "uz": f"Assalomu alaykum! Loyiha qoidasiga{rc['uz']} muvofiq mavjud o'yinchilar (jumladan ma'murlar va media-hamkorlar) nikiga o'xshash hamda haqoratli so'zlardan iborat nikneymlar taqiqlanadi.",
                    "en": f"Hello! Under project rules ({lbl_en}), nicknames similar to existing ones (including administrators and media partners) or containing insults/profanity are prohibited.",
                    "es": f"¡Hola! Según las reglas ({lbl_es}), están prohibidos los nombres similares a los de administradores o con insultos.",
                    "pt": f"Olá! Pelas regras ({lbl_pt}), são proibidos nicknames semelhantes aos de administradores ou com ofensas.",
                }
            elif cid0.startswith("GZ"):
                templates = {
                    "ru": f"Здравствуйте! По правилам Зелёной Зоны{rc['ru']}, в Green Zone (больницы, правительство, полиция, ФСБ, авторынок, СТО, казино и места работ) запрещены убийства, нанесение урона и криминальные действия.",
                    "kk": f"Сәлеметсіз бе! Жасыл аймақ ережесіне{rc['kk']} сәйкес Green Zone аумағында (аурухана, үкімет, полиция, ФСБ, автобазар, казино және жұмыс орындары) өлтіруге, зиян келтіруге және қылмыстық әрекеттерге тыйым салынады.",
                    "uz": f"Assalomu alaykum! Yashil zona qoidasiga{rc['uz']} muvofiq Green Zone hududida (shifoxona, hukumat, politsiya, FSB, avtobozor, kazino va ish joylari) o'ldirish, zarar yetkazish va jinoiy harakatlar taqiqlanadi.",
                    "en": f"Hello! Under Green Zone rules ({lbl_en}), killing, dealing damage, and criminal actions are strictly prohibited in Green Zones (hospitals, government, police, car market, casino, and job areas).",
                    "es": f"¡Hola! Según las reglas ({lbl_es}), están prohibidos los daños y delitos en las Zonas Verdes.",
                    "pt": f"Olá! Pelas regras ({lbl_pt}), são proibidos danos e ações criminosas nas Zonas Verdes.",
                }
            elif cid0.startswith("ЖБ"):
                templates = {
                    "ru": f"Здравствуйте! По правилам подачи жалоб{rc['ru']}, подать официальную жалобу на игрока или администратора можно в соответствующем разделе жалоб в лаунчере игры при наличии объективных доказательств нарушения.",
                    "kk": f"Сәлеметсіз бе! Шағым беру ережесіне{rc['kk']} сәйкес ойыншыға немесе әкімшіге ресми шағымды бұзушылықтың объективті дәлелдемелері болған жағдайда ойын лаунчеріндегі шағымдар бөлімі арқылы бере аласыз.",
                    "uz": f"Assalomu alaykum! Shikoyat topshirish qoidasiga{rc['uz']} muvofiq o'yinchi yoki ma'mur ustidan rasmiy shikoyatni qoidabuzarlikning xolis dalillari mavjud bo'lganda o'yin launcheridagi shikoyatlar bo'limi orqali yuborishingiz mumkin.",
                    "en": f"Hello! Under complaint rules ({lbl_en}), you can submit an official complaint against a player or administrator in the Complaints section of the game launcher when you have objective evidence.",
                    "es": f"¡Hola! Según las reglas ({lbl_es}), puede enviar una queja en el lanzador del juego adjuntando pruebas objetivas.",
                    "pt": f"Olá! Pelas regras ({lbl_pt}), você pode enviar uma reclamação no launcher do jogo anexando provas objetivas.",
                }
            else:
                templates = {
                    "ru": f"Здравствуйте! Вы обратились с вопросом по правилам проекта{rc['ru']} — ваш запрос передан администратору для подробного разъяснения.",
                    "kk": f"Сәлеметсіз бе! Сіз жоба ережесі бойынша сұрақ қойдыңыз{rc['kk']} — сұранысыңыз толық түсіндіру үшін әкімшіге берілді.",
                    "uz": f"Assalomu alaykum! Siz loyiha qoidasi bo'yicha savol yo'lladingiz{rc['uz']} — so'rovingiz batafsil tushuntirish uchun ma'murga yuborildi.",
                    "en": f"Hello! Regarding your question, please refer to ({lbl_en}). Your request has been forwarded to an administrator.",
                    "es": f"¡Hola! Con respecto a su pregunta, consulte ({lbl_es}). Su solicitud ha sido enviada a un administrador.",
                    "pt": f"Olá! Em relação à sua dúvida, consulte ({lbl_pt}). Seu pedido foi encaminhado a um administrador.",
                }
            return templates.get(reply_lang, templates["ru"]), cited_list
        templates = {
            "ru": "Здравствуйте! Ваш запрос по механике игры передан администратору для уточнения информации по вашему вопросу.",
            "kk": "Сәлеметсіз бе! Ойын механикасы бойынша сұрағыңыз ақпаратты нақтылау үшін әкімшіге берілді.",
            "uz": "Assalomu alaykum! O'yin mexanikasi bo'yicha savolingiz ma'lumotga aniqlik kiritish uchun ma'murga yuborildi.",
            "en": "Hello! Your request has been forwarded to an administrator for clarification.",
            "es": "¡Hola! Su solicitud ha sido enviada a un administrador para su aclaración.",
            "pt": "Olá! Seu pedido foi encaminhado a um administrador para esclarecimento.",
        }
        return templates.get(reply_lang, templates["ru"]), []

    # 7. report.offtopic: sent to human (needs_human=True), reminds rule 3.1 and asks to clarify question/complaint
    if category == "report.offtopic":
        r_en = f" ({lbl_en})" if lbl_en else ""
        r_es = f" ({lbl_es})" if lbl_es else ""
        r_pt = f" ({lbl_pt})" if lbl_pt else ""
        templates = {
            "ru": f"Здравствуйте! Ваш запрос передан администратору — пожалуйста, уточните ваш вопрос или суть проблемы. Напоминаем, что по правилам репорта{rc['ru']} запрещено отправлять оффтоп, флудить и просить телепорт или имущество.",
            "kk": f"Сәлеметсіз бе! Сұранысыңыз әкімшіге берілді — сұрағыңызды немесе мәселенің мәнін нақтылап жазуыңызды сұраймыз. Репорт ережесіне{rc['kk']} сәйкес оффтоп пен флуд жіберуге тыйым салынатынын еске саламыз.",
            "uz": f"Assalomu alaykum! So'rovingiz ma'murga yuborildi — iltimos, savolingizni yoki muammo mazmunini aniqlashtiring. Report qoidasiga{rc['uz']} muvofiq offtop va flud yuborish taqiqlanishini eslatib o'tamiz.",
            "en": f"Hello! Your request has been forwarded to an administrator — please clarify your question or issue. Please remember that under the report rules{r_en}, offtopic messages and flooding are prohibited.",
            "es": f"¡Hola! Su solicitud ha sido enviada a un administrador; por favor aclare su consulta{r_es}.",
            "pt": f"Olá! Seu pedido foi encaminhado a um administrador; por favor esclareça sua dúvida{r_pt}.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 8. report.complaint: urgent in-game violation report -> passed to admin, recommends launcher complaint if evidence exists
    if category == "report.complaint":
        r_ru = f" (вы указали {lbl_ru} правил)" if lbl_ru else ""
        r_kk = f" (сіз ереженің {lbl_kk_acc} көрсеттіңіз)" if lbl_kk_acc else ""
        r_uz = f" (siz {lbl_uz_acc} ko'rsatdingiz)" if lbl_uz_acc else ""
        r_en = f" (you specified {lbl_en})" if lbl_en else ""
        templates = {
            "ru": f"Здравствуйте! Ваш запрос о возможном нарушении в игре{nick_ru}{r_ru} передан администратору. При наличии скриншота или видеозаписи вы также можете оформить жалобу в лаунчере игры, указав никнейм нарушителя и сервер.",
            "kk": f"Сәлеметсіз бе! Ойындағы ықтимал тәртіп бұзушылық туралы сұранысыңыз{nick_kk}{r_kk} әкімшіге берілді. Скриншот немесе бейнежазба болса, тәртіп бұзушының никнеймі мен серверді көрсетіп, лаунчерде шағым қалдыра аласыз.",
            "uz": f"Assalomu alaykum! O'yindagi ehtimoliy qoidabuzarlik haqidagi so'rovingiz{nick_uz}{r_uz} ma'murga yuborildi. Skrinshot yoki videoyozuv mavjud bo'lsa, qoidabuzarning niki va serverni ko'rsatib, o'yin launcherida shikoyat qoldirishingiz mumkin.",
            "en": f"Hello! Your request regarding an in-game violation{nick_en}{r_en} has been forwarded to an administrator. If you have evidence, you may also submit a complaint in the game launcher.",
            "es": "¡Hola! Su solicitud sobre una infracción en el juego ha sido enviada a un administrador.",
            "pt": "Olá! Seu pedido sobre uma infração no jogo foi encaminhado a um administrador.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 9. complaint.other -> DO NOT claim the topic of the player's rule was violated; say 'вы указали пункт X правил'
    # Deterministic variants by (has_nickname, has_evidence, has_cited, has_server)
    if category == "complaint.other":
        c_ru = f" (вы указали {lbl_ru} правил)" if lbl_ru else ""
        c_kk = f" (сіз ереженің {lbl_kk_acc} көрсеттіңіз)" if lbl_kk_acc else ""
        c_uz = f" (siz {lbl_uz_acc} ko'rsatdingiz)" if lbl_uz_acc else ""
        c_en = f" (you specified {lbl_en} of the rules)" if lbl_en else ""
        c_es = f" (usted indicó la {lbl_es})" if lbl_es else ""
        c_pt = f" (você indicou a {lbl_pt})" if lbl_pt else ""

        fup = _build_complaint_followup(nickname, has_evidence, server, safe_nick_uz)
        templates = {
            "ru": f"Здравствуйте! Ваше обращение о возможном нарушении правил игры{nick_ru}{c_ru} зарегистрировано.{fup['ru']}",
            "kk": f"Сәлеметсіз бе! Ойын ережесінің ықтимал бұзылуы бойынша шағымыңыз{nick_kk}{c_kk} тіркелді.{fup['kk']}",
            "uz": f"Assalomu alaykum! O'yin qoidasining ehtimoliy buzilishi bo'yicha shikoyatingiz{nick_uz}{c_uz} ro'yxatga olindi.{fup['uz']}",
            "en": f"Hello! Your complaint{nick_en}{c_en} has been registered, and the attached materials have been forwarded to an administrator for review." if has_evidence else f"Hello! Your complaint{nick_en}{c_en} has been forwarded to an administrator. To review your inquiry, please attach a screenshot or video recording of the incident.",
            "es": f"¡Hola! Su queja{c_es} ha sido registrada y los materiales adjuntos han sido enviados a un administrador para su revisión." if has_evidence else f"¡Hola! Su queja{c_es} ha sido enviada a un administrador. Para revisarla, por favor adjunte una captura de pantalla o grabación de video.",
            "pt": f"Olá! Sua reclamação{c_pt} foi registrada e os materiais anexados foram encaminhados a um administrador para análise." if has_evidence else f"Olá! Sua reclamação{c_pt} foi encaminhada a um administrador. Para analisá-la, por favor anexe uma captura de tela ou gravação de vídeo.",
        }
        return templates.get(reply_lang, templates["ru"]), cited_list

    # 10. Category-specific complaint templates (complaint.insult, complaint.dm, complaint.cheat, complaint.scam, complaint.rmt_ads)
    # Deterministic sub-scenario selection based on player text and cited rules
    r_en = f" ({lbl_en})" if lbl_en else ""
    r_es = f" ({lbl_es})" if lbl_es else ""
    r_pt = f" ({lbl_pt})" if lbl_pt else ""

    if category == "complaint.insult":
        if FAMILY_INSULT_REGEX.search(p_text) or any(c in ("2.4", "2.6") for c in cited_list):
            desc_ru = f"возможного оскорбления или упоминания родных в чате{rc['ru']}"
            desc_kk = f"ойын чатында жақындарыңызға тіл тигізу немесе қорлау дерегі бойынша{rc['kk']}"
            desc_uz = f"o'yin chatida yaqinlarni tilga olish yoki haqorat qilish holati bo'yicha{rc['uz']}"
        else:
            desc_ru = f"возможного оскорбления или грубого общения в чате{rc['ru']}"
            desc_kk = f"ойын чатындағы ықтимал қорлау немесе әдепсіз сөздер бойынша{rc['kk']}"
            desc_uz = f"o'yin chatidagi ehtimoliy haqorat yoki nomaqbul muomala bo'yicha{rc['uz']}"
        desc_en = f"chat insults or inappropriate communication{r_en}"
        desc_es = f"insultos o comunicación inapropiada{r_es}"
        desc_pt = f"insultos ou comunicação inadequada{r_pt}"
    elif category == "complaint.dm":
        if any(c.startswith("GZ") for c in cited_list) or GZ_TEXT_REGEX.search(p_text):
            desc_ru = f"возможного нанесения урона или нарушения правил Зелёной Зоны (Green Zone){rc['ru']}"
            desc_kk = f"Жасыл аймақ (Green Zone) ережесін ықтимал бұзу немесе зиян келтіру бойынша{rc['kk']}"
            desc_uz = f"Yashil zona (Green Zone) qoidasini ehtimoliy buzish yoki zarar yetkazish bo'yicha{rc['uz']}"
        elif NONRP_CUFFS_DM_REGEX.search(p_text):
            desc_ru = f"возможного NonRP-задержания или применения наручников без причины{rc['ru']}"
            desc_kk = f"негізсіз ұстау немесе ықтимал NonRP әрекеттер бойынша{rc['kk']}"
            desc_uz = f"asossiz ushlab turish yoki ehtimoliy NonRP harakatlar bo'yicha{rc['uz']}"
        elif DB_VEHICLE_DM_REGEX.search(p_text):
            desc_ru = f"возможного наезда транспортом (DB) или нанесения урона без причины{rc['ru']}"
            desc_kk = f"көлікпен қасақана басу (DB) немесе себепсіз зиян келтіру бойынша{rc['kk']}"
            desc_uz = f"transportda qasddan bosish (DB) yoki sababsiz zarar yetkazish bo'yicha{rc['uz']}"
        elif "5.4" in cited_list or SK_TK_DM_REGEX.search(p_text):
            desc_ru = f"возможного убийства на спавне или урона по союзникам (SK / TK){rc['ru']}"
            desc_kk = f"спавнда өлтіру немесе одақтасқа зиян келтіру (SK / TK) бойынша{rc['kk']}"
            desc_uz = f"spavnda o'ldirish yoki ittifoqdoshga zarar yetkazish (SK / TK) bo'yicha{rc['uz']}"
        else:
            desc_ru = f"возможной стрельбы или убийства без игровой причины (DM){rc['ru']}"
            desc_kk = f"ойыншыға себепсіз зиян келтіру немесе өлтіру (DM) бойынша{rc['kk']}"
            desc_uz = f"o'yinchiga sababsiz zarar yetkazish yoki o'ldirish (DM) bo'yicha{rc['uz']}"
        desc_en = f"unprovoked damage or killing (DM / Green Zone){r_en}"
        desc_es = f"daño o muerte sin motivo (DM / Zona Verde){r_es}"
        desc_pt = f"dano ou morte sem motivo (DM / Zona Verde){r_pt}"
    elif category == "complaint.cheat":
        if BUG_ABUSE_CHEAT_REGEX.search(p_text):
            desc_ru = f"подозрения на использование игровых уязвимостей (багоюз){rc['ru']}"
            desc_kk = f"ойын қателіктерін (багоюз) пайдалану күдігі бойынша{rc['kk']}"
            desc_uz = f"o'yin xatolaridan (baglardan) foydalanish gumoni bo'yicha{rc['uz']}"
        elif AIM_MACRO_CHEAT_REGEX.search(p_text):
            desc_ru = f"подозрения на использование аимбота или сторонних скриптов{rc['ru']}"
            desc_kk = f"аимбот немесе бөгде скрипттер қолдану күдігі бойынша{rc['kk']}"
            desc_uz = f"aimbot yoki taqiqlangan skriptlardan foydalanish gumoni bo'yicha{rc['uz']}"
        else:
            desc_ru = f"подозрения на использование стороннего ПО (читов){rc['ru']}"
            desc_kk = f"бөгде бағдарлама (чит) қолдану күдігі бойынша{rc['kk']}"
            desc_uz = f"taqiqlangan dasturlar (chit) ishlatish gumoni bo'yicha{rc['uz']}"
        desc_en = f"suspected cheating or bug abuse{r_en}"
        desc_es = f"sospecha de uso de trampas o abuso de errores{r_es}"
        desc_pt = f"suspeita de uso de trapaças ou abuso de bugs{r_pt}"
    elif category == "complaint.scam":
        if "4.2" in cited_list or EXTORTION_SCAM_REGEX.search(p_text):
            desc_ru = f"возможного вымогательства имущества или шантажа{rc['ru']}"
            desc_kk = f"мүлікті ықтимал бопсалау дерегі бойынша{rc['kk']}"
            desc_uz = f"mulkni ehtimoliy tovlamachilik yo'li bilan talab qilish bo'yicha{rc['uz']}"
        else:
            desc_ru = f"возможного обмана при сделке или обмене имуществом{rc['ru']}"
            desc_kk = f"мүлік алмасу немесе мәміле кезіндегі ықтимал алаяқтық бойынша{rc['kk']}"
            desc_uz = f"mulk ayirboshlash yoki savdo vaqtidagi ehtimoliy firibgarlik bo'yicha{rc['uz']}"
        desc_en = f"suspected trade scam or extortion{r_en}"
        desc_es = f"posible estafa en intercambio o extorsión{r_es}"
        desc_pt = f"possível golpe em troca ou extorsão{r_pt}"
    else:  # complaint.rmt_ads
        if "2.3" in cited_list or ADS_RMT_REGEX.search(p_text):
            desc_ru = f"возможной сторонней рекламы в игровом чате{rc['ru']}"
            desc_kk = f"ойын чатындағы ықтимал бөгде жарнама бойынша{rc['kk']}"
            desc_uz = f"o'yin chatidagi ehtimoliy tashqi reklama bo'yicha{rc['uz']}"
        else:
            desc_ru = f"возможной продажи игровой валюты или имущества за реальные деньги{rc['ru']}"
            desc_kk = f"ойын валютасын немесе мүлікті нақты ақшаға сату күдігі бойынша{rc['kk']}"
            desc_uz = f"o'yin valyutasi yoki mulkni real pulga sotish gumoni bo'yicha{rc['uz']}"
        desc_en = f"suspected real-money trading (RMT) or unauthorized advertising{r_en}"
        desc_es = f"posible venta por dinero real o publicidad{r_es}"
        desc_pt = f"possível venda por dinheiro real ou publicidade{r_pt}"

    fup = _build_complaint_followup(nickname, has_evidence, server, safe_nick_uz)

    templates = {
        "ru": f"Здравствуйте! Ваша жалоба{nick_ru} по поводу {desc_ru} зарегистрирована.{fup['ru']}",
        "kk": f"Сәлеметсіз бе! Сіздің {desc_kk} шағымыңыз{nick_kk} тіркелді.{fup['kk']}",
        "uz": f"Assalomu alaykum! Sizning {desc_uz} shikoyatingiz{nick_uz} ro'yxatga olindi.{fup['uz']}",
        "en": f"Hello! Your complaint{nick_en} regarding {desc_en} has been registered.{fup['en']}",
        "es": f"¡Hola! Su queja sobre {desc_es} ha sido registrada.{fup['es']}",
        "pt": f"Olá! Sua reclamação sobre {desc_pt} foi registrada.{fup['pt']}",
    }
    return templates.get(reply_lang, templates["ru"]), cited_list


def build_fallback_response(
    player_text: str,
    kind: str,
    server_lang: Optional[str],
    reported_nick: Optional[str],
    rule_refs_raw: Optional[Any],
    attachments: Optional[int],
    images_count: Optional[int],
    is_spam_or_dup: bool,
    rag_items: List[RuleItem],
    is_error_fallback: bool = True,
    verdict: Optional[str] = None,
) -> TriageResponse:
    norm_text = normalize_text(player_text)
    lang = detect_language(norm_text)
    reply_lang = resolve_reply_lang(lang, server_lang)
    rule_refs = extract_rule_refs(norm_text, rule_refs_raw)
    has_ev = check_has_evidence(attachments, images_count, norm_text)
    nickname = extract_accused_nickname(norm_text, reported_nick)
    server = extract_server_from_text(norm_text)
    category = infer_category(
        norm_text,
        kind,
        rule_refs,
        has_ev,
        is_spam=is_spam_or_dup,
        verdict=verdict,
        rule_refs_raw=rule_refs_raw,
    )
    priority = compute_priority(category, has_ev, norm_text)

    cited_ids = select_best_citations(category, rule_refs, rag_items, has_ev, text=norm_text, verdict=verdict)
    draft_text, cited = build_safe_draft_text(
        category=category,
        reply_lang=reply_lang,
        nickname=nickname,
        cited_id=cited_ids,
        has_evidence=has_ev,
        rag_items=rag_items,
        verdict=verdict,
        player_text=norm_text,
        player_rule_refs=rule_refs,
        server=server,
    )

    if is_error_fallback:
        needs_human = True
        confidence = 0.35
        reasons = [
            f"правило-ориентированный fallback для категории {category}",
            "требуется проверка человеком из-за недоступности или ошибки LLM",
        ]
    else:
        confidence = calibrate_confidence(
            text=norm_text,
            lang=lang,
            category=category,
            rule_refs=rule_refs,
            cited_id=cited_ids[0] if cited_ids else None,
            has_evidence=has_ev,
            is_spam_or_dup=is_spam_or_dup,
        )
        needs_human = compute_needs_human(category, norm_text, confidence, rag_has_answer=bool(cited), verdict=verdict)
        reasons = [
            f"определена категория {category} по тексту и контексту правил",
            "решение о наказании или проверка остаётся за администратором" if needs_human else "содержательный ответ подтверждён базой правил",
        ]

    return TriageResponse(
        lang=lang,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        is_spam_or_duplicate=is_spam_or_dup,
        extracted=ExtractedData(
            nickname=nickname,
            rule_refs=rule_refs,
            has_evidence=has_ev,
            server=server,
        ),
        draft_reply=DraftReply(
            lang=reply_lang,  # type: ignore[arg-type]
            text=draft_text,
            cited=cited,
        ),
        needs_human=needs_human,
        confidence=confidence,
        reasons=reasons,
    )
