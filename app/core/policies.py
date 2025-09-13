from pydantic import BaseModel, field_validator


class DisclosureRequirements(BaseModel):
    trust_ge: float | None = None
    exact_question: bool | None = None


class DisclosureRules(BaseModel):
    full_on_valid_question: bool = True
    partial_if_low_trust: bool = True
    min_trust_for_gated: float = 0.4

    @field_validator('min_trust_for_gated')
    @classmethod
    def validate_min_trust(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError('min_trust_for_gated must be between 0 and 1')
        return v


class DistortionRules(BaseModel):
    enabled: bool = True
    by_defense: dict[str, float] = {}

    @field_validator('by_defense')
    @classmethod
    def validate_by_defense(cls, v: dict[str, float]) -> dict[str, float]:
        for key, value in v.items():
            if not (0 <= value <= 1):
                raise ValueError(f'by_defense[{key}] must be between 0 and 1, got {value}')
        return v


class RiskProtocol(BaseModel):
    trigger_keywords: list[str] = ["суицид", "убить себя", "не хочу жить"]
    response_style: str = "stable"
    lock_topics: list[str] = []


class StyleProfile(BaseModel):
    register: str = "colloquial"
    tempo: str = "medium"
    length: str = "short"


class Policies(BaseModel):
    disclosure_rules: DisclosureRules
    distortion_rules: DistortionRules
    risk_protocol: RiskProtocol
    style_profile: StyleProfile


def gated_access_allowed(trust: float, min_trust: float) -> bool:
    """Проверяет разрешен ли доступ к gated контенту"""
    return trust >= min_trust


def is_risk_trigger(text: str, keywords: list[str]) -> bool:
    """Проверяет содержит ли текст ключевые слова риска"""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def effective_disclosure_level(trust: float, rules: DisclosureRules) -> str:
    """Возвращает уровень раскрытия: 'full' | 'partial' | 'none'"""
    if trust >= rules.min_trust_for_gated:
        return "full"
    elif trust < rules.min_trust_for_gated and rules.partial_if_low_trust:
        return "partial"
    else:
        return "none"