import re
from typing import Optional
from app.normalize import UKRAINIAN_MARKER_REGEX, normalize_text

# Strictly Kazakh-specific Cyrillic letters (excluding 'і' which is also used in Ukrainian)
KK_STRICT_CHARS = set("қғңәөұүһҚҒҢӘӨҰҮҺ")

# Vocabulary markers per language (lowercase)
KK_WORDS = {
    "сәлеметсіз", "шағым", "шағымым", "тіркеу", "көмектесіңіз", "көмектесіңізші",
    "алдап", "кетті", "ойында", "ойынға", "мені", "менің", "маған", "саған",
    "және", "бірақ", "болады", "ақша", "ақшамды", "мүлкім", "жаза", "әкімші",
    "тармағы", "тармағының", "қорлады", "балағаттады", "көлік", "себепсіз",
    "өлтірді", "жасыл", "аймақ", "аймақта", "кіре", "алмай", "жатырмын",
    "бұзып", "түспеді", "теңге", "сұраймын", "қате", "беріңізші", "қалай",
    "өзгертуге", "деген", "ішінде", "тұрып", "қалдым", "отбасымды", "қолданып",
    "жүр", "орынға", "кірген", "сәтте", "түрмеден", "шықтым", "ештеңе",
    "жасамасамда", "болғам", "қарасам", "далада", "тастап", "осы", "адам",
    "осындай", "керек", "күнге", "үшін", "бул", "жапты", "жаба", "алмайды", "ғой",
}

UZ_WORDS = {
    "assalomu", "alaykum", "assalomaleykum", "asslom", "alekum", "salom",
    "murojaat", "murojaatingiz", "akkaunt", "akkauntimni", "parol", "parolim",
    "kira", "olmayapman", "ishlamayapti", "yordam", "bering", "berilar",
    "beringlar", "berila", "beradi", "beryabdi", "beryapti", "berdi", "bermoqchi",
    "qoida", "qoidani", "buzdi", "buzishdi", "mashina", "mashinam", "moshina",
    "moshinam", "moshinadan", "aldab", "ketdi", "ketishdi", "skrin", "skrinshot",
    "skrinlar", "yashil", "zonada", "zona", "uchun", "lekin", "tushmadi",
    "donat", "qildim", "haqorat", "haqoratli", "xaqorat", "qildi", "qilip",
    "qilib", "qimagan", "qilmadim", "qilish", "qilishmi", "qilaman", "meni",
    "menga", "manga", "mani", "oilamni", "onadan", "sabab", "sababsiz",
    "besabab", "otib", "bo'yicha", "boʻyicha", "tekshiring", "almashishda",
    "ilova", "qilindi", "qanday", "qanaqa", "o'zgartirish", "mumkin", "nima",
    "ichiga", "tushib", "qoldi", "chiqarib", "yuboring", "yubordi", "hisobimga",
    "yechib", "olindi", "iltimos", "chora", "korila", "kurilar", "koring",
    "korilsin", "nega", "bilan", "blan", "soz", "sozlar", "sokinmasam",
    "sukindi", "sokdi", "soʻkinish", "soʻkindi", "sokish", "sokingan",
    "uldirdi", "uldrdi", "uldryabdi", "oldirdi", "oʻldirdi", "kishanlab",
    "kishan", "sorayman", "sureman", "soriman", "bolasi", "nisbatan", "javob",
    "savol", "ozi", "ozidan", "uzidan", "uzi", "oʻzi", "oʻzidan", "boʻlaman",
    "boʻlinglar", "uyushtirdi", "jamodoshini", "quydi", "qoʻydi", "blmedi",
    "va", "deb", "emas", "emasman", "yoq", "odam", "bola", "knopkasi",
}

ES_WORDS = {
    "hola", "soporte", "pago", "cuenta", "tarjeta", "monedas", "hice", "ayuda",
    "servidor", "jugador", "favor", "gracias", "puedo", "problema", "recibí",
    "recibi", "hora", "estoy", "atrapado", "debajo", "mapa", "por", "con", "una",
    "no", "mi", "las", "el", "la", "en", "regla", "infracción", "queja",
}

PT_WORDS = {
    "olá", "ola", "meu", "minha", "personagem", "ficou", "preso", "debaixo",
    "mapa", "preciso", "teleporte", "ajuda", "pagamento", "conta", "não", "nao",
    "jogo", "obrigado", "perto", "hospital", "fazer", "cartão", "cartao",
    "moedas", "recebi", "servidor", "jogador", "usou", "regra", "no", "violação",
    "reclamação", "você",
}

EN_WORDS = {
    "hello", "hi", "help", "please", "account", "payment", "bought", "coins",
    "money", "charged", "did", "not", "arrive", "server", "player", "cannot",
    "can't", "login", "log", "hacked", "password", "reset", "someone", "profile",
    "card", "minutes", "ago", "my", "the", "and", "in", "to", "after", "with",
    "banned", "unban", "cheat", "cheater", "killed", "scammed",
}

LAT_RU_WORDS = {
    "zdravstvuyte", "privet", "pomogite", "pozhaluysta", "pojalusta", "pzh",
    "zastryal", "teksturah", "zaspavnite", "menya", "pomenyat", "nik", "igre",
    "pravilam", "skrin", "skriny", "admin", "adminy", "razbante", "vernite",
    "akkaunt", "dengi", "obmanul", "ubil", "chiter", "mashinu", "servere",
    "zhaloba", "oskorbil", "rodnyh", "pojaluysta", "kak", "chto", "gde", "kogda",
    "osk", "ock", "oski", "rod", "sogl", "fraps", "chek", "logi",
}

RU_WORDS = {
    "здравствуйте", "привет", "помогите", "пожалуйста", "застрял", "текстурах",
    "заспавните", "меня", "поменять", "ник", "игре", "правилам", "скрины",
    "скриншоты", "админ", "администратор", "разбаньте", "верните", "аккаунт",
    "деньги", "обманул", "убил", "читер", "машину", "сервере", "жалоба",
    "оскорблял", "оскорбил", "родных", "семью", "чате", "взломали", "задонатил",
    "рублей", "пришёл", "пришел", "донат", "не", "на", "по", "за", "из", "от",
    "ты", "домой", "когда", "давай", "бегом", "тут", "дед", "котом", "злые",
    "уже", "вместо", "того", "чтоб", "свою", "работу", "помогать", "нами",
    "прошу", "посадили", "без", "где", "как",
}


def _strip_nicknames_and_tokens(text: str) -> str:
    cleaned = re.sub(r"<(?:URL|HANDLE)>|https?://\S+", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[A-Za-z0-9]+_[A-Za-z0-9_]+\b", " ", cleaned)
    cleaned = re.sub(r"\b(?:GZ|ЖБ)\s*\d+(?:\.\d+)*\b", " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def detect_language(text: Optional[str]) -> str:
    norm = normalize_text(text)
    if not norm:
        return "und"

    cleaned = _strip_nicknames_and_tokens(norm)
    letters = [ch for ch in cleaned if ch.isalpha()]
    if not letters:
        return "und"

    tokens = re.findall(
        r"[a-zA-Zа-яА-ЯёЁқғңәөұүіһҚҒҢӘӨҰҮІҺєіїґЄІЇҐáéíóúüñ¿¡ãõçêôğ'‘’ʼ`ʻ]+",
        cleaned.lower(),
    )
    if not tokens:
        return "und"

    cyrillic_count = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
    latin_count = sum(
        1 for ch in letters if ("a" <= ch.lower() <= "z") or ch.lower() in "áéíóúüñãõçêôğ"
    )
    kk_strict_char_count = sum(1 for ch in letters if ch in KK_STRICT_CHARS)

    kk_hits = sum(1 for t in tokens if t in KK_WORDS)
    uz_hits = sum(1 for t in tokens if t in UZ_WORDS)
    es_hits = sum(1 for t in tokens if t in ES_WORDS)
    pt_hits = sum(1 for t in tokens if t in PT_WORDS)
    en_hits = sum(1 for t in tokens if t in EN_WORDS)
    lat_hits = sum(1 for t in tokens if t in LAT_RU_WORDS)
    ru_hits = sum(1 for t in tokens if t in RU_WORDS)

    uz_apostrophe = len(
        re.findall(r"\b[a-z]*(?:o['‘’ʼ`ʻ]|g['‘’ʼ`ʻ]|e['‘’ʼ`ʻ]|ğ)[a-z]*\b", cleaned.lower())
    )
    uz_suffix = len(
        re.findall(
            r"\b[a-zʻʼ'‘’`]{3,}(?:yapman|yapsiz|yapti|yabdi|votdi|voti|moqda|moqchi|madi|madim|magan|maganman|inglar|ilar|larizni|ishingizni)\b",
            cleaned.lower(),
        )
    )

    if cyrillic_count >= latin_count:
        # If the text contains Ukrainian words and zero strictly-Kazakh characters (қғңәөұүһ),
        # treat as Cyrillic 'ru' (even if Ukrainian 'мені' overlaps with Kazakh 'мені').
        if UKRAINIAN_MARKER_REGEX.search(cleaned) and kk_strict_char_count == 0:
            return "ru"
        if kk_hits >= 1 or kk_strict_char_count >= 2 or (kk_strict_char_count == 1 and ru_hits == 0):
            return "kk"
        return "ru"

    if (
        uz_apostrophe >= 1
        or uz_suffix >= 1
        or uz_hits >= 2
        or (uz_hits >= 1 and uz_hits >= max(en_hits, es_hits, pt_hits, lat_hits))
    ):
        return "uz"

    pt_chars = sum(1 for ch in cleaned.lower() if ch in "ãõçêô")
    es_chars = sum(1 for ch in cleaned.lower() if ch in "ñ¿¡")

    if pt_chars >= 1 or (pt_hits >= 2 and pt_hits > es_hits and pt_hits >= en_hits):
        return "pt"
    if es_chars >= 1 or (es_hits >= 2 and es_hits >= pt_hits and es_hits >= en_hits):
        return "es"

    translit_digraphs = len(
        re.findall(r"(?:zh|kh|ts|ch|sh|shch|yu|ya|y[aeou]|ost|iy|ogo|emu|eniy)", cleaned.lower())
    )
    if lat_hits >= 1 and lat_hits >= en_hits:
        return "lat"
    if translit_digraphs >= 2 and en_hits == 0 and uz_hits == 0 and es_hits == 0 and pt_hits == 0:
        return "lat"

    if en_hits >= 1:
        return "en"

    if uz_hits > 0:
        return "uz"
    if pt_hits > 0:
        return "pt"
    if es_hits > 0:
        return "es"
    if lat_hits > 0:
        return "lat"

    return "en"


def resolve_reply_lang(detected_lang: str, server_lang: Optional[str] = "ru") -> str:
    valid_reply_langs = {"ru", "kk", "uz", "en", "es", "pt"}
    if detected_lang in valid_reply_langs:
        return detected_lang
    if detected_lang == "lat":
        return "ru"
    if server_lang and server_lang.lower() in valid_reply_langs:
        return server_lang.lower()
    return "ru"
