"""
News Sentiment Analyzer for Crypto Trading Bot
Analyzes sentiment of crypto news articles
"""
import logging
from typing import Dict, List

logger = logging.getLogger('bot.news_sentiment_analyzer')


class NewsSentimentAnalyzer:
    """
    Analyzes sentiment of crypto news articles

    Uses keyword-based sentiment analysis for now.
    Can be extended with ML models in the future.
    """

    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.bullish_keywords = [
            'bullish', 'rally', 'surge', 'pump', 'moon', 'breakout',
            'breakthrough', 'all-time high', 'ath', 'gains', 'soar',
            'upgrade', 'adoption', 'partnership', 'integration', 'positive',
            'growth', 'rise', 'increase', 'profit', 'success'
        ]

        self.bearish_keywords = [
            'bearish', 'crash', 'dump', 'plunge', 'fall', 'decline',
            'drop', 'correction', 'selloff', 'sell-off', 'fear', 'panic',
            'hack', 'scam', 'fraud', 'regulatory', 'ban', 'crackdown',
            'concerns', 'negative', 'loss', 'losses', 'fails'
        ]

        logger.info("NewsSentimentAnalyzer initialized")

    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of given text

        Args:
            text: News article title or content

        Returns:
            Dictionary with sentiment analysis results
        """
        if not text:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}

        text_lower = text.lower()

        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)

        total_count = bullish_count + bearish_count

        if total_count == 0:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}

        score = (bullish_count - bearish_count) / total_count
        confidence = min(total_count / 5.0, 1.0)

        if score > 0.2:
            sentiment = 'bullish'
        elif score < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'bullish_keywords': bullish_count,
            'bearish_keywords': bearish_count
        }

    def analyze_news_batch(self, news_items: List[Dict]) -> Dict[str, any]:
        """
        Analyze sentiment of multiple news items

        Args:
            news_items: List of news items with 'title' and optionally 'content'

        Returns:
            Aggregate sentiment analysis
        """
        if not news_items:
            return {
                'overall_sentiment': 'neutral',
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'average_score': 0.0
            }

        sentiments = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        scores = []

        for item in news_items:
            text = item.get('title', '') + ' ' + item.get('content', '')
            result = self.analyze_sentiment(text)
            sentiments.append(result)
            scores.append(result['score'])

            if result['sentiment'] == 'bullish':
                bullish_count += 1
            elif result['sentiment'] == 'bearish':
                bearish_count += 1
            else:
                neutral_count += 1

        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score > 0.1:
            overall = 'bullish'
        elif avg_score < -0.1:
            overall = 'bearish'
        else:
            overall = 'neutral'

        return {
            'overall_sentiment': overall,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'average_score': avg_score,
            'total_analyzed': len(news_items),
            'individual_sentiments': sentiments
        }
