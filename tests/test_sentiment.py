"""
Tests for SentimentAnalyzer.
"""
from sentiment import SentimentAnalyzer
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def cfg():
    return MagicMock()


@pytest.fixture
def sa(cfg):
    return SentimentAnalyzer(cfg)


class TestSentimentAnalyzerInit:
    def test_running_is_false(self, sa):
        assert sa.running is False

    def test_has_keyword_sets(self, sa):
        assert 'bullish' in sa.bullish_keywords
        assert 'bearish' in sa.bearish_keywords


class TestAnalyzeTexts:
    def test_empty_list_returns_neutral(self, sa):
        result = sa.analyze_texts([])
        assert result['score'] == 0.5
        assert result['sentiment'] == 'neutral'
        assert result['fomo'] is False
        assert result['fud'] is False

    def test_bullish_keywords_give_bullish_score(self, sa):
        texts = ['bitcoin moon bullish pump rally buy hodl']
        result = sa.analyze_texts(texts)
        assert result['score'] > 0.5
        assert result['sentiment'] == 'bullish'
        assert result['bullish_count'] > 0

    def test_bearish_keywords_give_bearish_score(self, sa):
        texts = ['crash dump bear sell rekt fud fear']
        result = sa.analyze_texts(texts)
        assert result['score'] < 0.5
        assert result['sentiment'] == 'bearish'
        assert result['bearish_count'] > 0

    def test_neutral_text_gives_neutral_sentiment(self, sa):
        texts = ['the weather is nice today in the park']
        result = sa.analyze_texts(texts)
        assert result['sentiment'] == 'neutral'

    def test_very_bullish_triggers_fomo(self, sa):
        # Overwhelmingly bullish text
        texts = [' '.join(['moon', 'bullish', 'pump', 'rally', 'buy', 'hodl', 'ath', 'rocket'] * 5)]
        result = sa.analyze_texts(texts)
        assert result['fomo'] is True

    def test_very_bearish_triggers_fud(self, sa):
        texts = [' '.join(['crash', 'dump', 'bear', 'sell', 'rekt', 'fud', 'fear', 'panic'] * 5)]
        result = sa.analyze_texts(texts)
        assert result['fud'] is True

    def test_multiple_texts_aggregated(self, sa):
        texts = ['bullish moon pump', 'bearish dump crash']
        result = sa.analyze_texts(texts)
        assert result['bullish_count'] > 0
        assert result['bearish_count'] > 0

    def test_score_between_zero_and_one(self, sa):
        texts = ['random text with some words']
        result = sa.analyze_texts(texts)
        assert 0.0 <= result['score'] <= 1.0

    def test_total_words_counted(self, sa):
        texts = ['hello world foo bar']
        result = sa.analyze_texts(texts)
        assert result['total_words'] == 4

    def test_case_insensitive(self, sa):
        result_lower = sa.analyze_texts(['bullish'])
        result_upper = sa.analyze_texts(['BULLISH'])
        assert result_lower['bullish_count'] == result_upper['bullish_count']


class TestAnalyzeSingleText:
    def test_delegates_to_analyze_texts(self, sa):
        result = sa.analyze_single_text('moon bullish pump')
        assert isinstance(result, dict)
        assert 'score' in result

    def test_empty_string(self, sa):
        result = sa.analyze_single_text('')
        assert result['score'] == 0.5


@pytest.mark.asyncio
class TestSentimentAnalyzerLifecycle:
    async def test_start_sets_running(self, sa):
        await sa.start()
        assert sa.running is True
        await sa.stop()

    async def test_stop_clears_running(self, sa):
        await sa.start()
        await sa.stop()
        assert sa.running is False
