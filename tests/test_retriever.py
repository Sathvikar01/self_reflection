"""Tests for knowledge retriever module."""

import pytest
from src.knowledge.retriever import (
    KnowledgeRetriever,
    RetrievedFact,
    SCIENTIFIC_FACTS,
    FACTUAL_KEYWORDS,
)


class TestRetrievedFact:
    """Tests for RetrievedFact dataclass."""

    def test_creation(self):
        fact = RetrievedFact(
            topic="chemistry",
            question_pattern=r"diamond.*burn",
            fact="Diamonds can burn.",
            source="Materials science",
            confidence=1.0,
        )
        assert fact.topic == "chemistry"
        assert fact.fact == "Diamonds can burn."
        assert fact.confidence == 1.0

    def test_default_confidence(self):
        fact = RetrievedFact(
            topic="test",
            question_pattern=r"test",
            fact="Test fact",
            source="Test",
        )
        assert fact.confidence == 1.0


class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever class."""

    @pytest.fixture
    def retriever(self):
        return KnowledgeRetriever()

    def test_init_default(self, retriever):
        assert retriever.knowledge_base == SCIENTIFIC_FACTS

    def test_init_custom(self):
        custom = {"test": RetrievedFact("t", r"test", "f", "s")}
        retriever = KnowledgeRetriever(knowledge_base=custom)
        assert retriever.knowledge_base == custom

    def test_is_factual_question_true(self, retriever):
        assert retriever.is_factual_question("Can diamonds burn?") is True
        assert retriever.is_factual_question("Do fish drown?") is True
        assert retriever.is_factual_question("Is the earth flat?") is True

    def test_is_factual_question_false(self, retriever):
        assert retriever.is_factual_question("Tell me a joke") is False

    def test_is_factual_question_with_keywords(self, retriever):
        assert retriever.is_factual_question("diamond burning point") is True
        assert retriever.is_factual_question("fish drowning mechanism") is True

    def test_retrieve_relevant_facts_diamond(self, retriever):
        facts = retriever.retrieve_relevant_facts("Can diamonds burn in fire?")
        assert len(facts) >= 1
        assert any("diamond" in f.fact.lower() or "carbon" in f.fact.lower() for f in facts)

    def test_retrieve_relevant_facts_fish(self, retriever):
        facts = retriever.retrieve_relevant_facts("Can fish drown?")
        assert len(facts) >= 1

    def test_retrieve_relevant_facts_none(self, retriever):
        facts = retriever.retrieve_relevant_facts("What is the meaning of life?")
        assert len(facts) == 0

    def test_get_fact_by_topic(self, retriever):
        fact = retriever.get_fact_by_topic("chemistry")
        assert fact is not None
        assert fact.topic == "chemistry"

    def test_get_fact_by_topic_missing(self, retriever):
        assert retriever.get_fact_by_topic("nonexistent") is None

    def test_inject_knowledge_into_prompt(self, retriever):
        base_prompt = "Solve this problem."
        injected, facts = retriever.inject_knowledge_into_prompt(
            "Can diamonds burn?", base_prompt
        )
        assert len(facts) >= 1
        assert "RELEVANT SCIENTIFIC FACTS" in injected
        assert base_prompt in injected

    def test_inject_knowledge_no_match(self, retriever):
        base_prompt = "Solve this problem."
        injected, facts = retriever.inject_knowledge_into_prompt(
            "What is the meaning of life?", base_prompt
        )
        assert len(facts) == 0
        assert injected == base_prompt

    def test_inject_knowledge_max_facts(self, retriever):
        base_prompt = "Solve this problem."
        _, facts = retriever.inject_knowledge_into_prompt(
            "Can diamonds burn?", base_prompt, max_facts=1
        )
        assert len(facts) <= 1

    def test_get_all_topics(self, retriever):
        topics = retriever.get_all_topics()
        assert len(topics) > 0
        assert "chemistry" in topics

    def test_get_all_facts(self, retriever):
        facts = retriever.get_all_facts()
        assert len(facts) == len(SCIENTIFIC_FACTS)

    def test_create_reasoning_context_factual(self, retriever):
        ctx = retriever.create_reasoning_context("Can diamonds burn?")
        assert ctx["is_factual"] is True
        assert ctx["has_knowledge"] is True
        assert len(ctx["facts"]) > 0

    def test_create_reasoning_context_non_factual(self, retriever):
        ctx = retriever.create_reasoning_context("Tell me a story")
        assert ctx["has_knowledge"] is False

    def test_format_knowledge_for_prompt(self, retriever):
        facts = [
            RetrievedFact("t", r"p", "Fact one.", "s"),
            RetrievedFact("t", r"p", "Fact two.", "s"),
        ]
        formatted = retriever._format_knowledge_for_prompt(facts)
        assert "1. Fact one." in formatted
        assert "2. Fact two." in formatted
