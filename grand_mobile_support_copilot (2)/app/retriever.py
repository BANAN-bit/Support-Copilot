import math
import re
from collections import Counter
from typing import Dict, List, Optional
from app.rules_index import RuleItem, RulesIndex

MULTILINGUAL_KB_EXPANSIONS: Dict[str, str] = {
    "1.4": "взлом аккаунта украли аккаунт пароль привязка пропажа имущества бұзып кетті кіре алмай мүлкім жоғалды akkauntimni buzishdi parol hacked account stolen lost property",
    "1.6": "смена ника как поменять ник имя персонажа паспорт донат нон рп ник nik o'zgartirish никнейм өзгерту nickname change format",
    "2.1": "оскорбление мат токсичность чат оск қорлады балағаттады боқтық haqorat qildi sokdi insult toxicity oskorbil упоминание родных оск родных маму семью отбасымды анамды oilamni onamni",
    "2.2": "пропаганда наркотики экстремизм",
    "2.3": "реклама пиар промокод телеграм дискорд жарнама reklama",
    "2.4": "розжиг нация расизм религия политика ұлтшылдық millatchilik racism",
    "2.5": "музыка в микрофон саундпад шумы крики",
    "2.6": "угрозы в жизни доксинг деанон перенос конфликта ooc ic",
    "3.1": "застрял текстуры баг спавн телепорт репорт оффтоп провалился под карту переверните машину эвакуатор текстураның ішінде тұрып қалдым спавн беріңізші tekstura ichiga tushib qoldi yordam stuck textures teleport spawn bug offtopic zastryal",
    "3.4": "жалоба на админа администратора апелляция бана разбан пересмотр наказания не согласен с баном әкімші қате бан жазамды қайта қарау admin ban appeal",
    "4.1": "обман скам кинул на машину дом обмен имущество развод махинации алдап кетті көлік ауыстырғанда алаяқтық aldab ketdi mashina almashishda firibgarlik scam trade fraud obmanul",
    "4.2": "вымогательство угрозы ультиматум имущество",
    "4.4": "продажа покупка виртов за реал реальные деньги рмт rmt real money",
    "5.1": "нонрп поведение злоупотребление механиками nonrp mg pg",
    "5.2": "читы читер софт спидхак аим телепорт бот багоюз макросы автокликер чит қолданып ұшып жүр speedhack aimbot cheat hack",
    "5.3": "уход от рп дм дб убийство без причины задавил машиной стрельба себепсіз атып өлтірді sababsiz otib o'ldirdi dm db killed without reason ubil",
    "5.4": "тк ск спавнкилл тимкилл убийство у дома на спавне tk sk spawnkill teamkill",
    "GZ": "зелёная зона гз зз дм в гз убийство в зелёной зоне жасыл аймақ yashil zona green zone",
    "GZ 1.1": "зелёная зона гз убийство урон в зелёной зоне больница авторынок мэрия казино жасыл аймақ деген не yashil zona nima green zone",
    "GZ 1.2": "криминальные действия в зелёной зоне похищение ограбление верёвка мешок гз зз жасыл аймақта yashil zonada gz",
    "ЖБ 1.1.1": "как подать жалобу как залить жб куда писать жалобу раздел жалоб в лаунчере официальный инструмент",
    "ЖБ 1.1.2": "не по форме раздел жалоб не предназначен для вопросов общения технических проблем проверяю форму",
    "ЖБ 1.2.2": "нет доказательств без скринов без видео скриншоты отсутствуют обрезаны дәлелдеме жоқ skrinshot yo'q no evidence",
    "ЖБ 1.2.7": "текст жалобы содержит оскорбления мат токсичное поведение",
}

CATEGORY_TO_RULES: Dict[str, List[str]] = {
    "complaint.insult": ["2.1", "2.4", "2.6"],
    "complaint.cheat": ["5.2"],
    "complaint.dm": ["5.3", "5.4", "GZ", "GZ 1.1", "GZ 1.2"],
    "complaint.scam": ["4.1", "4.2"],
    "complaint.rmt_ads": ["2.3", "4.4"],
    "complaint.admin": ["3.4"],
    "complaint.other": ["5.1", "1.6", "5.5"],
    "complaint.invalid": ["ЖБ 1.1.2", "ЖБ 1.2.7", "ЖБ 1.2.2"],
    "report.question": ["1.6", "ЖБ 1.1.1", "GZ 1.1", "5.3", "3.1"],
    "report.help": ["3.1"],
    "report.complaint": ["3.1", "5.3", "2.1", "5.2"],
    "report.account": ["1.4", "1.7"],
    "report.payment": ["1.4", "4.1"],
    "report.offtopic": ["3.1"],
}

GZ_QUERY_REGEX = re.compile(
    r"\b(?:gz|гз|зз|zz|зелён\w*\s+зон\w*|зелен\w*\s+зон\w*|yashil\s+zona\w*|жасыл\s+аймақ\w*|green\s+zone)\b",
    re.IGNORECASE,
)


def _tokenize_for_bm25(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁқғңәөұүіһҚҒҢӘӨҰҮІҺ0-9\.]+", text.lower())
    tokens: List[str] = []
    for w in words:
        if len(w) >= 2:
            tokens.append(w)
            if len(w) >= 5 and not w[0].isdigit():
                tokens.append(w[:4])
                tokens.append(w[:5])
    return tokens


class BM25Retriever:
    """
    BM25 retriever over rules.json items (sections 30-32) + multilingual Knowledge Base expansions.
    Returns 3-5 relevant rule items considering player rule_refs and category_hint.
    """

    def __init__(self, rules_index: Optional[RulesIndex] = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.rules_index = rules_index or RulesIndex()
        self.k1 = k1
        self.b = b
        self.items: List[RuleItem] = self.rules_index.all_items()
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.avgdl: float = 1.0
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        self.doc_tokens.clear()
        self.doc_freqs.clear()
        self.doc_lens.clear()

        for item in self.items:
            kb_extra = MULTILINGUAL_KB_EXPANSIONS.get(item.id, "")
            full_doc = f"{item.id} {item.section_title} {item.text} {kb_extra}"
            tokens = _tokenize_for_bm25(full_doc)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            for tok in set(tokens):
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1

        if self.doc_lens:
            self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        else:
            self.avgdl = 1.0

    def retrieve(
        self,
        query: str,
        rule_refs: Optional[List[str]] = None,
        category_hint: Optional[str] = None,
        has_evidence: bool = True,
        kind: str = "report",
        top_k: int = 4,
        verdict: Optional[str] = None,
    ) -> List[RuleItem]:
        top_k = max(3, min(5, top_k))
        selected_ids: List[str] = []
        selected_items: List[RuleItem] = []

        cat_candidates = list(CATEGORY_TO_RULES.get(category_hint or "", []))
        if category_hint == "complaint.dm" and GZ_QUERY_REGEX.search(query or ""):
            cat_candidates = ["GZ", "GZ 1.1", "5.3", "5.4", "GZ 1.2"]
        if (verdict or "").strip().lower() == "invalid_format" or category_hint == "complaint.invalid":
            cat_candidates = ["ЖБ 1.1.2", "ЖБ 1.2.7", "ЖБ 1.2.2"]

        for hr in cat_candidates[:2]:
            item = self.rules_index.get(hr)
            if item and item.id not in selected_ids and len(selected_items) < top_k:
                selected_ids.append(item.id)
                selected_items.append(item)

        if rule_refs:
            for ref in rule_refs:
                item = self.rules_index.get(ref)
                if item and item.id not in selected_ids and len(selected_items) < top_k:
                    selected_ids.append(item.id)
                    selected_items.append(item)

        q_tokens = _tokenize_for_bm25(query)
        n_docs = len(self.items)
        scores: Dict[str, float] = {}
        hint_rules = set(cat_candidates)

        for idx, item in enumerate(self.items):
            score = 0.0
            doc_toks = self.doc_tokens[idx]
            doc_len = self.doc_lens[idx]
            tf_counter = Counter(doc_toks)

            for qt in q_tokens:
                df = self.doc_freqs.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_counter.get(qt, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * ((tf * (self.k1 + 1.0)) / denom)

            if item.id in hint_rules:
                score += 3.5

            if kind == "complaint" and (not has_evidence or (verdict or "").strip().lower() == "invalid_format") and item.id in ("ЖБ 1.1.2", "ЖБ 1.2.7"):
                score += 2.5

            scores[item.id] = score

        ranked = sorted(self.items, key=lambda x: scores.get(x.id, 0.0), reverse=True)
        for item in ranked:
            if len(selected_items) >= top_k:
                break
            if item.id not in selected_ids:
                if scores.get(item.id, 0.0) > 0 or len(selected_items) < 3:
                    selected_ids.append(item.id)
                    selected_items.append(item)

        return selected_items[:5]

    @staticmethod
    def format_context(items: List[RuleItem]) -> str:
        if not items:
            return "Нет релевантных пунктов правил."
        return "\n".join(f"- {it.id}: {it.text}" for it in items)
