import asyncio
import logging
from typing import List, Dict
import re

logger = logging.getLogger('bot.sentiment')


class SentimentAnalyzer:
    """
    Basic sentiment analyzer using keyword-based approach
    Can be extended with BERT/Transformers for more advanced analysis
    """

    def __init__(self, cfg, redis_url=None):
        self.cfg = cfg
        self.redis_url = redis_url
        self.running = False

        # Keyword dictionaries for basic sentiment analysis
        self.bullish_keywords = {
            'bullish', 'bull', 'moon', 'pump', 'rally', 'breakout', 'surge',
            'green', 'profit', 'gains', 'up', 'high', 'ath', 'buy', 'long',
            'rocket', 'lambo', 'hodl', 'accumulate', 'bullrun', 'optimistic'
        }

        self.bearish_keywords = {
            'bearish', 'bear', 'dump', 'crash', 'sell', 'short',
            'red', 'loss', 'down', 'low', 'dip', 'drop', 'fall', 'panic',
            'rekt', 'liquidation', 'fud', 'fear', 'pessimistic', 'warning'
        }

    async def start(self):
        """Start sentiment analysis loop"""
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Sentiment analyzer started")

    async def stop(self):
        """Stop sentiment analysis"""
        self.running = False
        # Cancel the task to prevent "Task was destroyed" warning
        if hasattr(self, '_task') and self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Sentiment analyzer stopped")

    async def _loop(self):
        """Main sentiment analysis loop"""
        while self.running:
            try:
                # In a real implementation, this would:
                # 1. Poll social APIs (Twitter/Reddit via Tweepy/PRAW)
                # 2. Get news from NewsAPI
                # 3. Run sentiment analysis on collected texts
                # 4. Store results in Redis

                # For now, just log that we're running
                logger.debug("Sentiment analysis loop running...")
                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Error in sentiment loop: {e}")
                await asyncio.sleep(60)

    def analyze_texts(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment of text list using keyword matching

        Args:
            texts: List of text strings to analyze

        Returns:
            Dict with sentiment score and signals
        """
        if not texts:
            return {'score': 0.5, 'sentiment': 'neutral', 'fomo': False, 'fud': False}

        bullish_count = 0
        bearish_count = 0
        total_words = 0

        for text in texts:
            # Convert to lowercase and split into words
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)

            # Count bullish and bearish keywords
            for word in words:
                if word in self.bullish_keywords:
                    bullish_count += 1
                elif word in self.bearish_keywords:
                    bearish_count += 1

        # Calculate sentiment score (0 = very bearish, 1 = very bullish)
        if total_words == 0:
            score = 0.5
        else:
            # Weight by keyword density
            bullish_density = bullish_count / max(total_words, 1)
            bearish_density = bearish_count / max(total_words, 1)

            # Normalize to 0-1 range
            total_density = bullish_density + bearish_density
            if total_density > 0:
                score = bullish_density / total_density
            else:
                score = 0.5

        # Determine sentiment label
        if score > 0.65:
            sentiment = 'bullish'
        elif score < 0.35:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        return {
            'score': round(score, 3),
            'sentiment': sentiment,
            'fomo': score > 0.7,  # Fear of missing out (very bullish)
            'fud': score < 0.3,   # Fear, uncertainty, doubt (very bearish)
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total_words': total_words
        }

    def analyze_single_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        return self.analyze_texts([text])
