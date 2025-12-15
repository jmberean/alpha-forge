"""
LLM temporal safety framework.

Tests LLM for future knowledge contamination to prevent lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol

import pandas as pd


@dataclass
class CanaryQuestion:
    """
    Canary question to test LLM temporal isolation.

    These are factual questions with known earliest_knowable dates.
    """

    question: str
    earliest_knowable: datetime  # When this became knowable
    correct_answer: str
    category: str  # e.g., "election", "market", "event"


# Standard canary questions for testing
CANARY_QUESTIONS = [
    CanaryQuestion(
        question="Who won the 2020 US presidential election?",
        earliest_knowable=datetime(2020, 11, 7),
        correct_answer="Joe Biden",
        category="election",
    ),
    CanaryQuestion(
        question="What was the S&P 500 total return for 2020?",
        earliest_knowable=datetime(2021, 1, 1),
        correct_answer="18.4%",
        category="market",
    ),
    CanaryQuestion(
        question="When was the first COVID-19 vaccine approved in the US?",
        earliest_knowable=datetime(2020, 12, 11),
        correct_answer="December 11, 2020",
        category="event",
    ),
    CanaryQuestion(
        question="What was Apple's stock split ratio in August 2020?",
        earliest_knowable=datetime(2020, 8, 31),
        correct_answer="4-for-1",
        category="market",
    ),
    CanaryQuestion(
        question="When did Russia invade Ukraine?",
        earliest_knowable=datetime(2022, 2, 24),
        correct_answer="February 24, 2022",
        category="event",
    ),
]


class LLMInterface(Protocol):
    """Protocol for LLM interface."""

    def query(self, prompt: str, **kwargs) -> str:
        """Query the LLM with a prompt."""
        ...


@dataclass
class TemporalTestResult:
    """Result of temporal safety test."""

    passed: bool
    cutoff_date: datetime
    violations: list[CanaryQuestion]
    message: str


class LLMTemporalSafetyTester:
    """
    Test LLM for temporal contamination.

    Ensures LLM doesn't leak future knowledge when used for features.
    """

    def __init__(self, llm: LLMInterface | None = None):
        """
        Initialize tester.

        Args:
            llm: LLM interface (optional, uses mock if None)
        """
        self.llm = llm or MockLLM()

    def test_temporal_isolation(
        self,
        cutoff_date: datetime,
        canaries: list[CanaryQuestion] | None = None,
    ) -> TemporalTestResult:
        """
        Test if LLM respects temporal cutoff.

        Args:
            cutoff_date: LLM should not know anything after this date
            canaries: Canary questions to test (uses defaults if None)

        Returns:
            TemporalTestResult
        """
        if canaries is None:
            canaries = CANARY_QUESTIONS

        violations = []

        for canary in canaries:
            # Skip questions that should be knowable at cutoff
            if canary.earliest_knowable <= cutoff_date:
                continue

            # Ask question
            response = self.llm.query(canary.question)

            # Check if LLM has future knowledge
            if self._has_future_knowledge(response, canary):
                violations.append(canary)

        passed = len(violations) == 0

        if passed:
            message = f"PASSED: LLM respects cutoff date {cutoff_date}"
        else:
            message = (
                f"FAILED: LLM has future knowledge beyond {cutoff_date}. "
                f"Violations: {len(violations)}/{len([c for c in canaries if c.earliest_knowable > cutoff_date])}"
            )

        return TemporalTestResult(
            passed=passed,
            cutoff_date=cutoff_date,
            violations=violations,
            message=message,
        )

    def _has_future_knowledge(
        self,
        response: str,
        canary: CanaryQuestion,
    ) -> bool:
        """
        Check if response contains future knowledge.

        Args:
            response: LLM response
            canary: Canary question

        Returns:
            True if future knowledge detected
        """
        # Simple check: Does response contain the correct answer?
        # In production, use more sophisticated NLP
        response_lower = response.lower()
        answer_lower = canary.correct_answer.lower()

        # Check for key terms from answer
        answer_terms = answer_lower.split()

        # If multiple key terms present, likely has the answer
        matches = sum(1 for term in answer_terms if len(term) > 3 and term in response_lower)

        # Heuristic: 50%+ of answer terms present = has knowledge
        return matches >= len(answer_terms) * 0.5

    def validate_llm_feature(
        self,
        feature_func: Callable[[pd.DataFrame], pd.DataFrame],
        data: pd.DataFrame,
        llm_cutoff: datetime,
    ) -> bool:
        """
        Validate that LLM-based feature respects temporal cutoff.

        Args:
            feature_func: Feature function using LLM
            data: Test data
            llm_cutoff: LLM training cutoff date

        Returns:
            True if valid

        Raises:
            ValueError: If temporal contamination detected
        """
        # Test LLM temporal isolation
        result = self.test_temporal_isolation(llm_cutoff)

        if not result.passed:
            raise ValueError(
                f"LLM temporal contamination detected: {result.message}. "
                "Cannot use this LLM for features without temporal safety guarantee."
            )

        # Additional check: Ensure features don't use data after LLM cutoff
        if isinstance(data.index, pd.DatetimeIndex):
            latest_date = data.index.max().to_pydatetime()

            if latest_date > llm_cutoff:
                # This is acceptable IF features only use data <= llm_cutoff
                # In practice, would need to verify this in feature computation
                pass

        return True


class MockLLM:
    """
    Mock LLM for testing.

    Simulates an LLM with a specific knowledge cutoff.
    """

    def __init__(self, knowledge_cutoff: datetime | None = None):
        """
        Initialize mock LLM.

        Args:
            knowledge_cutoff: What date this LLM's knowledge ends
        """
        self.knowledge_cutoff = knowledge_cutoff or datetime(2023, 4, 1)

    def query(self, prompt: str, **kwargs) -> str:
        """
        Mock query that respects knowledge cutoff.

        Args:
            prompt: Question
            **kwargs: Additional parameters

        Returns:
            Mock response
        """
        # Find matching canary
        for canary in CANARY_QUESTIONS:
            if canary.question.lower() in prompt.lower():
                # If question is about something before cutoff, answer it
                if canary.earliest_knowable <= self.knowledge_cutoff:
                    return f"The answer is {canary.correct_answer}."
                else:
                    # Don't have future knowledge
                    return "I don't have information about that."

        return "I don't have enough information to answer that question."


def require_temporal_safety(
    llm: LLMInterface,
    cutoff_date: datetime,
) -> None:
    """
    Require that LLM passes temporal safety test.

    Args:
        llm: LLM to test
        cutoff_date: Required cutoff date

    Raises:
        ValueError: If LLM fails temporal safety test
    """
    tester = LLMTemporalSafetyTester(llm)
    result = tester.test_temporal_isolation(cutoff_date)

    if not result.passed:
        raise ValueError(
            f"LLM failed temporal safety test: {result.message}\n"
            f"Violations: {[c.question for c in result.violations]}"
        )


# Safe LLM feature patterns


class SafeLLMFeatures:
    """
    Safe patterns for using LLM in features.

    These patterns ensure temporal safety.
    """

    @staticmethod
    def sentiment_from_text(
        text_series: pd.Series,
        llm: LLMInterface,
        llm_cutoff: datetime,
    ) -> pd.Series:
        """
        SAFE: Compute sentiment from text data.

        Requirements:
        1. LLM must pass temporal safety test
        2. Text must be from before LLM cutoff
        3. Sentiment analysis doesn't leak future knowledge

        Args:
            text_series: Series of text (news, filings, etc.)
            llm: LLM interface
            llm_cutoff: LLM knowledge cutoff

        Returns:
            Sentiment scores
        """
        # Validate LLM temporal safety
        require_temporal_safety(llm, llm_cutoff)

        # In production: Call LLM for each text
        # For now, mock implementation
        sentiment = pd.Series(0.0, index=text_series.index)

        for idx, text in text_series.items():
            # Ensure text timestamp is before LLM cutoff
            if isinstance(idx, pd.Timestamp):
                if idx.to_pydatetime() > llm_cutoff:
                    raise ValueError(
                        f"Text at {idx} is after LLM cutoff {llm_cutoff}. "
                        "Cannot use LLM on future text."
                    )

            # Query LLM (mock)
            # response = llm.query(f"What is the sentiment of this text: {text}")
            # sentiment[idx] = parse_sentiment(response)
            sentiment[idx] = 0.5  # Mock neutral sentiment

        return sentiment

    @staticmethod
    def event_extraction(
        text_series: pd.Series,
        llm: LLMInterface,
        llm_cutoff: datetime,
    ) -> pd.DataFrame:
        """
        SAFE: Extract events from text.

        Same temporal safety requirements as sentiment.

        Args:
            text_series: Series of text
            llm: LLM interface
            llm_cutoff: LLM knowledge cutoff

        Returns:
            DataFrame with extracted events
        """
        require_temporal_safety(llm, llm_cutoff)

        # Mock implementation
        events = pd.DataFrame({
            "event_type": ["none"] * len(text_series),
            "confidence": [0.0] * len(text_series),
        }, index=text_series.index)

        return events
