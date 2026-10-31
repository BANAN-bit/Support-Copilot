import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from app.normalize import normalize_text


def _parse_iso_time(ts: Optional[str]) -> Optional[float]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        clean = ts.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(clean).timestamp()
    except Exception:
        return None


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁқғңәөұүіһҚҒҢӘӨҰҮІҺ0-9]+", text.lower())


def _char_ngrams(text: str, n: int = 3) -> Counter:
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        return Counter([cleaned]) if cleaned else Counter()
    return Counter(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


def cosine_similarity_counter(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2:
        return 0.0
    common = set(c1.keys()) & set(c2.keys())
    num = sum(c1[k] * c2[k] for k in common)
    if num == 0:
        return 0.0
    den1 = math.sqrt(sum(v * v for v in c1.values()))
    den2 = math.sqrt(sum(v * v for v in c2.values()))
    if den1 == 0.0 or den2 == 0.0:
        return 0.0
    return num / (den1 * den2)


def minhash_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Set Jaccard over character n-grams (exact MinHash target)."""
    s1 = set(_char_ngrams(text1, n=n).keys())
    s2 = set(_char_ngrams(text2, n=n).keys())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(len(s1 | s2))


SUBSTANTIVE_KEYWORDS = {
    "бан", "мут", "чит", "дм", "дб", "ск", "тк", "гз", "gz", "жб", "обман",
    "скам", "аккаунт", "взлом", "пароль", "донат", "платеж", "оплата", "деньги",
    "застрял", "спавн", "телепорт", "текстур", "баг", "ник", "админ", "оскорб",
    "родн", "убил", "машин", "сервер", "вопрос", "помог", "көмек", "шағым",
    "қорла", "алдап", "бұзып", "төлем", "ақша", "yordam", "akkaunt", "parol",
    "donat", "aldab", "haqorat", "hacked", "payment", "stuck", "account", "pago",
}


def is_spam_or_flood_text(text: Optional[str]) -> bool:
    """Detect spam, flood, meaningless symbol strings, or all-caps without substance."""
    norm = normalize_text(text)
    if not norm:
        return True

    without_url = re.sub(r"<URL>|https?://\S+", "", norm, flags=re.IGNORECASE).strip()
    letters = [c for c in without_url if c.isalpha()]
    if len(letters) < 3:
        return True

    low = without_url.lower()
    has_substance = any(kw in low for kw in SUBSTANTIVE_KEYWORDS) or bool(
        re.search(r"\b\d+\.\d+\b", low)
    )

    if re.search(r"(.)\1{5,}", low) and not has_substance:
        return True
    if re.search(r"(?:ах|ха|ah|ha|пж|pj|лол|lol){3,}", low) and not has_substance:
        return True
    if re.search(r"\b(?:дайте\s+(?:миллион|денег|админку|лидерку)|просто\s+так\s+пж)", low):
        return True

    alpha_low = [c.lower() for c in letters]
    if len(alpha_low) >= 10 and (len(set(alpha_low)) / float(len(alpha_low))) < 0.22 and not has_substance:
        return True

    words = _word_tokens(without_url)
    if len(words) >= 4:
        most_common_word, count = Counter(words).most_common(1)[0]
        if count >= 4 and (count / float(len(words))) >= 0.5 and not has_substance:
            return True

    upper_count = sum(1 for c in letters if c.isupper())
    if len(letters) >= 12 and (upper_count / float(len(letters))) > 0.85 and not has_substance:
        return True

    return False


class Deduplicator:
    """
    Stateful deduplicator by reporter_hash + time window + text similarity (TF-IDF cosine / MinHash),
    plus bonus criterion (Section 7.4 & 13.1): 1 player, same issue/series, >= 5 tickets within 1 hour.
    """

    def __init__(
        self,
        time_window_sec: float = 3600.0 * 6,
        burst_window_sec: float = 3600.0,
        burst_limit_per_hour: int = 5,
        sim_threshold: float = 0.72,
        max_history_per_user: int = 50,
    ) -> None:
        self.time_window_sec = time_window_sec
        self.burst_window_sec = burst_window_sec
        self.burst_limit_per_hour = burst_limit_per_hour
        self.sim_threshold = sim_threshold
        self.max_history_per_user = max_history_per_user
        self._history: Dict[str, List[Tuple[Optional[float], str, Counter, Counter]]] = defaultdict(list)

    def reset(self) -> None:
        self._history.clear()

    @property
    def tracked_reporters(self) -> int:
        """Количество авторов, по которым хранится история (для GET /stats)."""
        return len(self._history)

    def check_and_record(
        self,
        text: str,
        reporter_hash: Optional[str] = None,
        created_at: Optional[str] = None,
        duplicate_hint: Optional[bool] = None,
    ) -> Tuple[bool, bool]:
        """Returns (is_spam_or_duplicate, is_duplicate_by_history)."""
        norm = normalize_text(text)
        spam_flag = is_spam_or_flood_text(norm)

        if duplicate_hint is True:
            return True, True

        dup_flag = False
        ts = _parse_iso_time(created_at)
        w_cnt = Counter(_word_tokens(norm))
        ng_cnt = _char_ngrams(norm, n=3)

        if reporter_hash and norm:
            prev_list = self._history[reporter_hash]
            recent_in_hour = 1  # including current ticket

            for prev_ts, prev_norm, prev_w, prev_ng in prev_list:
                if ts is not None and prev_ts is not None:
                    delta = abs(ts - prev_ts)
                    if delta <= self.burst_window_sec:
                        recent_in_hour += 1
                    if delta > self.time_window_sec:
                        continue
                else:
                    recent_in_hour += 1

                if norm == prev_norm:
                    dup_flag = True
                    break
                cos_w = cosine_similarity_counter(w_cnt, prev_w)
                cos_ng = cosine_similarity_counter(ng_cnt, prev_ng)
                jaccard = minhash_jaccard(norm, prev_norm, n=3)
                if max(cos_w, cos_ng, jaccard) >= self.sim_threshold:
                    dup_flag = True
                    break

            # Bonus criterion (Section 7.4 / 13.1): 5+ tickets in 1 hour from the same reporter
            if recent_in_hour >= self.burst_limit_per_hour:
                dup_flag = True

            prev_list.append((ts, norm, w_cnt, ng_cnt))
            if len(prev_list) > self.max_history_per_user:
                prev_list.pop(0)

        return (spam_flag or dup_flag), dup_flag
