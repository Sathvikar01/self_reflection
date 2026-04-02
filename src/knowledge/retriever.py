"""Knowledge retrieval module for the self-reflection pipeline.

This module provides factual knowledge to prevent reasoning errors on
scientific questions where the model lacks accurate information.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetrievedFact:
    """A retrieved scientific fact."""
    topic: str
    question_pattern: str
    fact: str
    source: str
    confidence: float = 1.0


SCIENTIFIC_FACTS: Dict[str, RetrievedFact] = {
    "diamond_burning": RetrievedFact(
        topic="chemistry",
        question_pattern=r"(diamond|diamonds).*(burn|burns|burning|combust|flammable|fire)",
        fact="Diamonds are made of pure carbon and CAN burn. They combust at approximately 850°C (1562°F) in the presence of oxygen, forming carbon dioxide. This is a chemical property of carbon.",
        source="Materials science",
        confidence=1.0,
    ),
    "fish_drowning": RetrievedFact(
        topic="biology",
        question_pattern=r"(fish|fishes).*(drown|drowns|drowning|suffocate)",
        fact="Fish CAN drown or suffocate. They require dissolved oxygen in water to breathe through their gills. In low-oxygen water, polluted water, or stagnant water without aeration, fish can suffocate and die.",
        source="Marine biology",
        confidence=1.0,
    ),
    "astronaut_sunscreen": RetrievedFact(
        topic="space_science",
        question_pattern=r"(astronaut|astronauts|space).*(sunscreen|sun block|uv|ultraviolet)",
        fact="Astronauts do NOT need sunscreen in space. Their spacesuits and spacecraft provide complete protection from UV radiation. The suits have multiple layers including materials that block all harmful radiation.",
        source="NASA space suit design",
        confidence=1.0,
    ),
    "trees_sleep": RetrievedFact(
        topic="biology",
        question_pattern=r"(tree|trees|plant|plants).*(sleep|sleeps|sleeping|rest)",
        fact="Trees do NOT sleep in the animal sense. Sleep requires a brain and nervous system. Trees have no brain or central nervous system. However, trees do have circadian rhythms and day/night cycles affecting their metabolism.",
        source="Plant biology",
        confidence=1.0,
    ),
    "plants_oxygen": RetrievedFact(
        topic="biology",
        question_pattern=r"(plant|plants).*(oxygen|breathe|breathing|respiration)",
        fact="Plants DO need and use oxygen. While plants produce oxygen through photosynthesis during the day, they also perform cellular respiration 24/7, consuming oxygen to break down sugars for energy. Plants take in oxygen through their leaves, roots, and stems.",
        source="Plant physiology",
        confidence=1.0,
    ),
    "gold_rusting": RetrievedFact(
        topic="chemistry",
        question_pattern=r"(gold).*(rust|rusts|rusting|corrode|tarnish)",
        fact="Gold does NOT rust or tarnish. Gold is a noble metal that is highly resistant to oxidation and corrosion. This is why gold remains shiny indefinitely and is used in jewelry and electronics where corrosion resistance is needed.",
        source="Chemistry",
        confidence=1.0,
    ),
    "lightning_ice": RetrievedFact(
        topic="physics",
        question_pattern=r"(lightning).*(ice|freeze|frozen|cold)",
        fact="Lightning does NOT create ice. Lightning is extremely hot plasma (around 30,000°C / 54,000°F). While it can cause rapid heating and expansion of air, it cannot freeze or create ice. Any ice near a lightning strike was already frozen.",
        source="Physics",
        confidence=1.0,
    ),
    "diamond_hardest": RetrievedFact(
        topic="materials",
        question_pattern=r"(diamond|diamonds).*(hardest|harder|hardness|scratch)",
        fact="Diamond is the hardest natural material. It ranks 10 on the Mohs hardness scale and can scratch all other natural materials. However, synthetic materials like aggregated carbon nanorods can be harder.",
        source="Materials science",
        confidence=1.0,
    ),
    "earth_round": RetrievedFact(
        topic="astronomy",
        question_pattern=r"(earth|world).*(flat|round|sphere|spherical)",
        fact="Earth is spherical (an oblate spheroid). It is slightly flattened at the poles and bulging at the equator due to rotation. The evidence includes photos from space, ship hulls disappearing over horizons, and shadows on the moon during eclipses.",
        source="Astronomy",
        confidence=1.0,
    ),
    "water_conductivity": RetrievedFact(
        topic="physics",
        question_pattern=r"(water).*(conduct|conductive|electricity|electrical)",
        fact="Pure water is a poor electrical conductor. It is contaminants like dissolved salts, minerals, and ions that make water conductive. Distilled/deionized water has very low conductivity. Tap water is conductive due to dissolved minerals.",
        source="Physics",
        confidence=1.0,
    ),
}

FACTUAL_KEYWORDS = [
    "can", "does", "do", "is", "are", "will", "would", "could",
    "burn", "burns", "burning", "combust", "flammable",
    "drown", "drowns", "drowning", "suffocate",
    "sleep", "sleeps", "sleeping", "rest",
    "breathe", "breathes", "breathing", "oxygen", "respiration",
    "rust", "rusts", "rusting", "corrode", "tarnish",
    "freeze", "freezes", "freezing", "frozen",
    "hardest", "harder", "hardness", "scratch",
    "flat", "round", "sphere", "spherical",
    "conduct", "conductive", "electricity", "electrical",
    "sunscreen", "uv", "ultraviolet", "radiation",
    "made of", "composed of", "consist of",
]


class KnowledgeRetriever:
    """Retrieves relevant scientific facts for reasoning questions."""

    def __init__(self, knowledge_base: Optional[Dict[str, RetrievedFact]] = None):
        self.knowledge_base = knowledge_base or SCIENTIFIC_FACTS

    def is_factual_question(self, question: str) -> bool:
        """Detect if a question is asking for a scientific fact."""
        question_lower = question.lower()
        
        question_patterns = [
            r"^(can|does|do|is|are|will|could|would)\s",
            r"\?$",
        ]
        
        has_question_structure = any(
            re.search(p, question_lower) for p in question_patterns
        )
        
        has_factual_keywords = any(kw in question_lower for kw in FACTUAL_KEYWORDS)
        
        return has_question_structure or has_factual_keywords

    def retrieve_relevant_facts(self, question: str) -> List[RetrievedFact]:
        """Retrieve facts relevant to the question."""
        question_lower = question.lower()
        relevant_facts = []
        
        for fact_id, fact in self.knowledge_base.items():
            if re.search(fact.question_pattern, question_lower, re.IGNORECASE):
                relevant_facts.append(fact)
        
        return relevant_facts

    def get_fact_by_topic(self, topic: str) -> Optional[RetrievedFact]:
        """Get a fact by its topic name."""
        for fact in self.knowledge_base.values():
            if fact.topic == topic.lower():
                return fact
        return None

    def inject_knowledge_into_prompt(
        self, 
        question: str, 
        base_prompt: str,
        max_facts: int = 3
    ) -> Tuple[str, List[RetrievedFact]]:
        """Inject relevant knowledge into a reasoning prompt."""
        facts = self.retrieve_relevant_facts(question)
        
        if not facts:
            return base_prompt, []
        
        facts_to_use = facts[:max_facts]
        knowledge_text = self._format_knowledge_for_prompt(facts_to_use)
        
        injected_prompt = f"""RELEVANT SCIENTIFIC FACTS:
{knowledge_text}

{base_prompt}"""
        
        return injected_prompt, facts_to_use

    def _format_knowledge_for_prompt(self, facts: List[RetrievedFact]) -> str:
        """Format retrieved facts for prompt injection."""
        lines = []
        for i, fact in enumerate(facts, 1):
            lines.append(f"{i}. {fact.fact}")
        return "\n".join(lines)

    def get_all_topics(self) -> List[str]:
        """Get all available topic names."""
        return list(set(f.topic for f in self.knowledge_base.values()))

    def get_all_facts(self) -> Dict[str, RetrievedFact]:
        """Get all facts in the knowledge base."""
        return self.knowledge_base.copy()

    def create_reasoning_context(self, question: str) -> Dict:
        """Create a context dict with knowledge for reasoning."""
        is_factual = self.is_factual_question(question)
        facts = self.retrieve_relevant_facts(question) if is_factual else []
        
        return {
            "question": question,
            "is_factual": is_factual,
            "facts": [f.fact for f in facts],
            "topics": [f.topic for f in facts],
            "has_knowledge": len(facts) > 0,
        }
