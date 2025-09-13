"""
Retrieve node for knowledge base fragment retrieval.
Filters fragments based on access permissions, topics, and adds noise for realism.
"""

import random
import logging

from sqlalchemy import select, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.core.tables import KBFragment

logger = logging.getLogger(__name__)


async def retrieve(
    db: AsyncSession,
    case_id: str, 
    intent: str, 
    topics: list[str], 
    session_state_compact: dict, 
    top_k: int = 3
) -> list[dict]:
    """
    Retrieve knowledge base fragments based on access permissions and topics.
    
    Args:
        db: AsyncSession for database operations
        case_id: UUID of the case
        intent: Intent from {open_question, clarify, risk_check, rapport}
        topics: List of topics from normalize (sleep, mood, alcohol, work, family)
        session_state_compact: Dict with fields trust, access_level, risk_status
        top_k: Maximum number of fragments to return (default 3)
    
    Returns:
        List of dicts with format:
        [
            {
                "id": str(fragment.id),
                "type": fragment.type,
                "text": fragment.text,
                "metadata": fragment.fragment_metadata
            }
        ]
    """
    try:
        # Get user trust level
        trust_level = session_state_compact.get("trust", 0.0)
        
        # Check if case exists
        case_check = await db.execute(
            text("SELECT id FROM cases WHERE id = :case_id"),
            {"case_id": case_id}
        )
        if not case_check.fetchone():
            return []
        
        # Build base query for accessible fragments
        query = select(KBFragment).where(KBFragment.case_id == case_id)
        
        # Apply availability filters
        availability_conditions = [
            KBFragment.availability == "public"
        ]
        
        # Add gated access if trust level is sufficient
        gated_condition = and_(
            KBFragment.availability == "gated",
            or_(
                # Fragment has no disclosure requirements
                ~KBFragment.fragment_metadata.has_key("disclosure_requirements"),
                # Fragment has disclosure requirements but trust meets threshold
                text("(metadata->>'disclosure_requirements')::jsonb->>'trust_ge' IS NULL"),
                text("CAST((metadata->>'disclosure_requirements')::jsonb->>'trust_ge' AS FLOAT) <= :trust_level")
            )
        )
        availability_conditions.append(gated_condition)
        
        # Apply availability filter (exclude hidden)
        query = query.where(or_(*availability_conditions))
        
        # Apply topic filter
        if topics:
            # Filter by topics using ORM
            query = query.where(KBFragment.fragment_metadata['topic'].astext.in_(topics))
        
        # Execute query with trust level parameter
        result = await db.execute(
            query.limit(top_k),
            {"trust_level": trust_level}
        )
        
        fragments = result.scalars().all()
        
        # Convert to return format
        retrieved_fragments = []
        for fragment in fragments:
            retrieved_fragments.append({
                "id": str(fragment.id),
                "type": fragment.type,
                "text": fragment.text,
                "metadata": fragment.fragment_metadata
            })
        
        # Add noise with 20% probability
        if random.random() < 0.2 and retrieved_fragments:
            noise_fragment = await _get_random_public_fragment(
                db, case_id, topics, trust_level
            )
            if noise_fragment and len(retrieved_fragments) < top_k:
                retrieved_fragments.append(noise_fragment)
        
        return retrieved_fragments[:top_k]
        
    except SQLAlchemyError as e:
        logger.exception("Database error in retrieve", case_id=case_id, error=str(e))
        return []
    except ValueError as e:
        logger.exception("Invalid case_id format", case_id=case_id, error=str(e))
        raise
    except Exception as e:
        logger.exception("Unexpected error in retrieve", case_id=case_id, error=str(e))
        raise


async def _get_random_public_fragment(
    db: AsyncSession, 
    case_id: str, 
    excluded_topics: list[str],
    trust_level: float
) -> dict | None:
    """
    Get a random public fragment that is NOT from the specified topics.
    
    Args:
        db: Database session
        case_id: UUID of the case
        excluded_topics: Topics to exclude from noise
        trust_level: Current trust level
        
    Returns:
        Fragment dict or None if no suitable fragment found
    """
    try:
        # Build query for public fragments not in excluded topics
        query = select(KBFragment).where(
            and_(
                KBFragment.case_id == case_id,
                KBFragment.availability == "public"
            )
        )
        
        # Exclude fragments from specified topics
        if excluded_topics:
            topic_exclusions = []
            for topic in excluded_topics:
                topic_exclusions.append(
                    text("metadata->>'topic' != :topic").bindparam(topic=topic)
                )
            query = query.where(and_(*topic_exclusions))
        
        # Get all matching fragments
        result = await db.execute(query, {"trust_level": trust_level})
        candidates = result.scalars().all()
        
        if not candidates:
            return None
            
        # Pick random fragment
        selected = random.choice(candidates)
        
        return {
            "id": str(selected.id),
            "type": selected.type,
            "text": selected.text,
            "metadata": selected.fragment_metadata
        }
        
    except SQLAlchemyError as e:
        logger.exception("Database error in _get_random_public_fragment", case_id=case_id, error=str(e))
        return None
    except Exception as e:
        logger.exception("Unexpected error in _get_random_public_fragment", case_id=case_id, error=str(e))
        return None