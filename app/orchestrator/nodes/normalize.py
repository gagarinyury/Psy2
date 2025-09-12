"""
Normalize node for therapist utterance analysis.
Extracts intent, topics, risk flags, and summary from therapist utterance.
"""

def normalize(therapist_utterance: str, session_state_compact: dict) -> dict:
    """
    Analyze therapist utterance and extract structured information.
    
    Args:
        therapist_utterance: The therapist's statement/question
        session_state_compact: Current session state (not used in current implementation)
    
    Returns:
        dict with keys:
            - intent: str from {open_question, clarify, risk_check, rapport}
            - topics: list[str] extracted topics
            - risk_flags: list[str] risk indicators
            - last_turn_summary: str truncated to 200 chars
    """
    utterance_lower = therapist_utterance.lower()
    
    # Extract intent based on simple rules
    intent = _extract_intent(utterance_lower)
    
    # Extract topics
    topics = _extract_topics(utterance_lower)
    
    # Check for risk flags
    risk_flags = _extract_risk_flags(utterance_lower)
    
    # Create summary
    last_turn_summary = _create_summary(therapist_utterance)
    
    return {
        "intent": intent,
        "topics": topics,
        "risk_flags": risk_flags,
        "last_turn_summary": last_turn_summary
    }


def _extract_intent(utterance_lower: str) -> str:
    """Extract intent based on keyword patterns."""
    
    # Risk check keywords
    risk_keywords = ["суицид", "убить себя", "не хочу жить", "покончить с жизнью"]
    if any(keyword in utterance_lower for keyword in risk_keywords):
        return "risk_check"
    
    # Clarify keywords  
    clarify_keywords = ["как", "что", "когда", "где", "почему", "какой"]
    if any(keyword in utterance_lower for keyword in clarify_keywords):
        return "clarify"
    
    # Rapport keywords
    rapport_keywords = ["понимаю", "сочувствую", "поддерживаю"]
    if any(keyword in utterance_lower for keyword in rapport_keywords):
        return "rapport"
    
    # Default
    return "open_question"


def _extract_topics(utterance_lower: str) -> list[str]:
    """Extract topics based on keyword matching."""
    topics = []
    
    # Topic keyword mappings
    topic_keywords = {
        "sleep": ["спать", "спите", "сон", "бессонница", "засыпа"],
        "mood": ["настроение", "депрессия", "грусть", "радость", "тревога"],
        "alcohol": ["алкоголь", "пить", "выпивка", "водка", "пиво"],
        "work": ["работа", "работой", "карьера", "коллеги", "босс"],
        "family": ["семья", "семьей", "родители", "дети", "жена", "муж"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in utterance_lower for keyword in keywords):
            topics.append(topic)
    
    return topics


def _extract_risk_flags(utterance_lower: str) -> list[str]:
    """Extract risk flags based on suicide-related keywords."""
    risk_flags = []
    
    suicide_keywords = [
        "суицид", "убить себя", "не хочу жить", 
        "покончить с жизнью", "повеситься", "отравиться"
    ]
    
    if any(keyword in utterance_lower for keyword in suicide_keywords):
        risk_flags.append("suicide_ideation")
    
    return risk_flags


def _create_summary(utterance: str) -> str:
    """Create summary by truncating to 200 characters."""
    if len(utterance) <= 200:
        return utterance
    
    return utterance[:200] + "..."