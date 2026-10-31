"""Pydantic-модели запроса и ответа сервиса.

Требуется Pydantic 2.x (см. requirements.txt): валидаторы описаны через
`field_validator`, лишние поля запроса отбрасываются через `extra="ignore"`.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompatModel(BaseModel):
    """Базовая модель запроса: игнорирует лишние поля входного JSON."""

    model_config = ConfigDict(extra="ignore")


AllowedLang = Literal["ru", "kk", "uz", "en", "es", "pt", "lat", "und"]
AllowedReplyLang = Literal["ru", "kk", "uz", "en", "es", "pt"]
AllowedKind = Literal["complaint", "report"]
AllowedPriority = Literal["p0", "p1", "p2", "p3"]

ALLOWED_CATEGORIES = (
    "complaint.insult",
    "complaint.cheat",
    "complaint.dm",
    "complaint.scam",
    "complaint.rmt_ads",
    "complaint.admin",
    "complaint.other",
    "complaint.invalid",
    "report.question",
    "report.help",
    "report.complaint",
    "report.account",
    "report.payment",
    "report.offtopic",
)

AllowedCategory = Literal[
    "complaint.insult",
    "complaint.cheat",
    "complaint.dm",
    "complaint.scam",
    "complaint.rmt_ads",
    "complaint.admin",
    "complaint.other",
    "complaint.invalid",
    "report.question",
    "report.help",
    "report.complaint",
    "report.account",
    "report.payment",
    "report.offtopic",
]


class MessageItem(BaseModel):
    role: str = Field(default="player", description="player | reported_player | admin")
    text: str = Field(default="", description="Message text")
    ts: Optional[str] = Field(default=None, description="Optional message timestamp")
    attachments: Optional[int] = Field(default=0, description="Optional attachments count on message (Section 8.1)")

    @field_validator("text", mode="before")
    @classmethod
    def ensure_text_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)

    @field_validator("role", mode="before")
    @classmethod
    def ensure_role_str(cls, v: Any) -> str:
        if not v:
            return "player"
        return str(v)

    @field_validator("attachments", mode="before")
    @classmethod
    def normalize_msg_attachments(cls, v: Any) -> int:
        if v is None:
            return 0
        if isinstance(v, list):
            return len(v)
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return 0


class TicketRequest(CompatModel):
    ticket_id: Optional[str] = None
    kind: AllowedKind = Field(default="report", description="complaint or report")
    messages: List[MessageItem] = Field(default_factory=list)
    attachments: Optional[int] = Field(default=0)
    images_count: Optional[int] = Field(default=0)
    reported_nick: Optional[str] = Field(default=None)
    reported_hash: Optional[str] = Field(default=None)
    rule_refs_raw: Optional[Any] = Field(default=None)
    server_lang: Optional[str] = Field(default="ru")
    region: Optional[str] = Field(default="cis")
    reporter_hash: Optional[str] = Field(default=None)
    created_at: Optional[str] = Field(default=None)
    status: Optional[int] = Field(default=None)
    verdict: Optional[str] = Field(default=None)
    punishment: Optional[str] = Field(default=None)
    violation: Optional[str] = Field(default=None)
    DUPLICATE_HINT: Optional[bool] = Field(default=None)
    duplicate_hint: Optional[bool] = Field(default=None)
    text: Optional[str] = Field(default=None)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, v: Any) -> str:
        if isinstance(v, str) and v.strip().lower() in ("complaint", "report"):
            return v.strip().lower()
        return "report"

    @field_validator("attachments", "images_count", mode="before")
    @classmethod
    def normalize_counts(cls, v: Any) -> int:
        if v is None:
            return 0
        if isinstance(v, list):
            return len(v)
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return 0


class ExtractedData(BaseModel):
    nickname: Optional[str] = None
    rule_refs: List[str] = Field(default_factory=list)
    has_evidence: bool = False
    server: Optional[str] = None


class DraftReply(BaseModel):
    lang: AllowedReplyLang = "ru"
    text: str
    cited: List[str] = Field(default_factory=list)


class TriageResponse(BaseModel):
    lang: AllowedLang
    category: AllowedCategory
    priority: AllowedPriority
    is_spam_or_duplicate: bool
    extracted: ExtractedData
    draft_reply: DraftReply
    needs_human: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str]

    @field_validator("reasons", mode="after")
    @classmethod
    def check_reasons(cls, v: Any) -> List[str]:
        items = [str(r).strip() for r in (v or []) if str(r).strip()]
        if not items:
            raise ValueError("reasons must contain at least one non-empty item")
        return items[:3]

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
