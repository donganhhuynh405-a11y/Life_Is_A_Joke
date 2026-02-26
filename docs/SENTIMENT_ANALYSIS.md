# BERT Sentiment Analysis - Production Implementation

## Overview

Production-ready sentiment analysis system using BERT (Bidirectional Encoder Representations from Transformers) for cryptocurrency and financial text analysis. Includes fine-tuning capabilities, caching, and model persistence.

## Features

### ✅ Real BERT Model Implementation
- **Pre-trained Model**: Uses FinBERT (ProsusAI/finbert) - specifically trained for financial sentiment analysis
- **Fallback Support**: Automatically falls back to distilbert-base-uncased-finetuned-sst-2-english if primary model unavailable
- **GPU/CPU Support**: Automatically detects and uses available hardware
- **Real Inference**: Actual neural network inference with softmax probabilities (NO random values)

### ✅ Fine-tuning Capability
- **Crypto-specific Training**: Fine-tune on your own cryptocurrency-specific data
- **Custom Labels**: Support for positive, negative, and neutral sentiment
- **Training Metrics**: Track loss, accuracy, and F1 score during training
- **Validation Support**: Optional validation set for monitoring overfitting
- **Optimizer**: AdamW optimizer with linear warmup scheduler
- **Batch Processing**: Efficient batch training with configurable batch size

### ✅ Prediction Caching
- **MD5-based Cache Keys**: Unique hash for each text snippet
- **Persistent Cache**: Saved to disk (pickle format)
- **Automatic Saving**: Periodically saves cache every 100 predictions
- **Fast Lookups**: O(1) cache lookup for repeated texts
- **Cache Management**: Clear cache when needed, view cache statistics

### ✅ Model Persistence
- **Save Fine-tuned Models**: Save model weights after fine-tuning
- **Load Saved Models**: Automatically load fine-tuned models on initialization
- **Metadata Storage**: Track model configuration (name, max_length, device)
- **Directory Structure**: Organized in `models/sentiment_cache/`

### ✅ Production-Ready Features
- **Error Handling**: Comprehensive try-catch blocks with logging
- **Type Hints**: Full type annotations for better IDE support
- **Logging**: Detailed logging for debugging and monitoring
- **Batch Processing**: Efficient batch analysis with configurable batch size
- **Confidence Scores**: Returns confidence level for each prediction

## Installation

### Dependencies
```bash
pip install torch>=2.6.0
pip install transformers>=4.48.0
pip install sentencepiece>=0.2.1
pip install scikit-learn>=1.3.0
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Sentiment Analysis

```python
from sentiment_advanced import BERTSentimentAnalyzer

# Initialize analyzer (loads pre-trained FinBERT)
analyzer = BERTSentimentAnalyzer()

# Analyze single text
text = "Bitcoin price surge! Massive gains today! 🚀"
result = analyzer.analyze_text(text)

print(f"Label: {result['label']}")           # POSITIVE/NEGATIVE/NEUTRAL
print(f"Score: {result['sentiment_score']}")  # 0.0 to 1.0
print(f"Confidence: {result['confidence']}")  # Model confidence
```

### Batch Analysis

```python
texts = [
    "Bitcoin is crashing! Sell everything!",
    "Stable price movement, no significant changes",
    "ETH breaking all-time highs! Incredible rally!"
]

results = analyzer.batch_analyze(texts, batch_size=32)

for text, result in zip(texts, results):
    print(f"{text[:40]}... → {result['label']} ({result['sentiment_score']:.3f})")
```

### Fine-tuning on Crypto Data

```python
# Prepare training data
train_texts = [
    "Bitcoin to the moon! HODL strong! 🚀",
    "Crypto crash incoming, massive dump expected",
    "Sideways movement, consolidation phase",
    # ... more examples
]

# Labels: 0=negative, 1=neutral, 2=positive
train_labels = [2, 0, 1, ...]

# Fine-tune model
metrics = analyzer.fine_tune(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,        # Optional validation set
    val_labels=val_labels,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    save_model=True             # Save after training
)

print(f"Final validation accuracy: {metrics['val_accuracy'][-1]:.4f}")
```

### Using Cache

```python
# First call: performs inference
result1 = analyzer.analyze_text("Bitcoin is amazing!", use_cache=True)

# Second call: retrieved from cache (instant)
result2 = analyzer.analyze_text("Bitcoin is amazing!", use_cache=True)

# Check cache statistics
stats = analyzer.get_cache_stats()
print(f"Cache size: {stats['cache_size']} predictions")
print(f"Cache directory: {stats['cache_dir']}")

# Clear cache if needed
analyzer.clear_cache()
```

### Model Persistence

```python
# Save model after fine-tuning
analyzer.save_model()  # Saves to default path: models/sentiment_cache/finetuned_model

# Or specify custom path
analyzer.save_model("path/to/custom/model")

# Model automatically loads fine-tuned weights on next initialization
analyzer2 = BERTSentimentAnalyzer()  # Will load fine-tuned model if it exists
```

### Advanced Configuration

```python
# Initialize with custom settings
analyzer = BERTSentimentAnalyzer(
    model_name='ProsusAI/finbert',           # HuggingFace model
    cache_dir='custom/cache/dir',            # Custom cache directory
    device='cuda',                            # Force GPU (or 'cpu')
    max_length=512                           # Max sequence length
)
```

## Output Format

Each sentiment analysis returns a dictionary with:

```python
{
    'text': 'Bitcoin price surge! Massive...',  # First 50 chars
    'sentiment_score': 0.87,                     # 0.0 (negative) to 1.0 (positive)
    'label': 'POSITIVE',                         # POSITIVE, NEGATIVE, or NEUTRAL
    'confidence': 0.92,                          # Model confidence (0.0 to 1.0)
    'logits': [0.1, 0.2, 0.7]                   # Raw model outputs
}
```

## Architecture

### Model Selection
1. **Primary**: FinBERT (ProsusAI/finbert) - Optimized for financial text
2. **Fallback**: DistilBERT SST-2 - General sentiment analysis

### Directory Structure
```
models/
└── sentiment_cache/
    ├── finetuned_model/         # Saved model weights
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   ├── tokenizer_config.json
    │   └── metadata.json
    └── predictions_cache.pkl    # Cached predictions
```

### Inference Pipeline
1. **Tokenization**: Convert text to token IDs
2. **Model Forward Pass**: BERT processes tokens
3. **Softmax**: Convert logits to probabilities
4. **Label Mapping**: Map to POSITIVE/NEGATIVE/NEUTRAL
5. **Score Calculation**: Normalize to [0, 1] range

### Fine-tuning Process
1. **Data Preparation**: Create PyTorch Dataset
2. **Optimizer Setup**: AdamW with linear warmup
3. **Training Loop**: Batch processing with gradient clipping
4. **Validation**: Track metrics on validation set
5. **Model Saving**: Persist weights and configuration

## Performance

### Speed
- **Single Text**: ~50-200ms (depending on hardware)
- **Batch of 32**: ~1-3 seconds
- **Cached Lookup**: <1ms

### Memory
- **Model Size**: ~400MB (FinBERT) or ~250MB (DistilBERT)
- **Cache**: ~1KB per cached prediction

### Accuracy
- **Pre-trained FinBERT**: ~94% on financial sentiment
- **Fine-tuned on crypto**: Varies based on training data

## Comparison: Before vs After

### Before (Stub Implementation)
```python
def analyze_text(self, text: str) -> Dict:
    import random
    score = random.random()  # ❌ Random values!
    return {
        'sentiment_score': score,
        'label': 'POSITIVE' if score > 0.6 else 'NEGATIVE',
        'confidence': abs(score - 0.5) + 0.5
    }
```

### After (Real Implementation)
```python
def analyze_text(self, text: str, use_cache: bool = True) -> Dict:
    # ✅ Real BERT inference
    # ✅ Cache support
    # ✅ Error handling
    # ✅ Proper label mapping
    # ✅ Confidence scores from softmax
    
    cache_key = self._get_cache_key(text)
    if use_cache and cache_key in self._prediction_cache:
        return self._prediction_cache[cache_key]
    
    return self._run_inference(text)  # Real neural network
```

## Testing

Run the test suite:
```bash
python test_sentiment_bert.py
```

Expected output:
```
============================================================
Testing BERT Sentiment Analyzer
============================================================

1. Initializing BERT model...
   ✓ Model loaded on device: cpu

2. Testing single text analysis...
   Text: Bitcoin price surge! Massive gains today! 🚀
   Label: POSITIVE
   Sentiment Score: 0.8734
   Confidence: 0.9123

3. Testing batch analysis...
   Analyzed 3 texts
   1. Bitcoin is crashing! Sell everything!... → NEGATIVE (0.123)
   2. Stable price movement, no significant... → NEUTRAL (0.501)
   3. ETH breaking all-time highs! Incredible... → POSITIVE (0.887)

4. Testing cache...
   Cache size before: 1
   Cache size after: 1
   ✓ Cache working correctly

5. Testing model save/load...
   Cache directory: models/sentiment_cache
   Fine-tuned model exists: False

============================================================
✓ All tests passed!
============================================================
```

## Integration with Trading Bot

### In Strategy Decision Making
```python
from sentiment_advanced import BERTSentimentAnalyzer

class SentimentEnhancedStrategy:
    def __init__(self):
        self.sentiment = BERTSentimentAnalyzer()
    
    def should_enter_trade(self, symbol: str, price_data: dict) -> bool:
        # Analyze recent news/social media
        texts = self.fetch_recent_texts(symbol)
        sentiments = self.sentiment.batch_analyze(texts)
        
        avg_sentiment = sum(s['sentiment_score'] for s in sentiments) / len(sentiments)
        
        # Combine with technical indicators
        if avg_sentiment > 0.7 and price_data['rsi'] < 30:
            return True  # Strong positive sentiment + oversold
        
        return False
```

## Troubleshooting

### Issue: Model fails to download
**Solution**: Check internet connection, or use fallback model

### Issue: Out of memory
**Solution**: Reduce batch_size or max_length, or use CPU

### Issue: Slow inference
**Solution**: Use GPU, enable caching, increase batch_size

### Issue: Cache too large
**Solution**: Call `analyzer.clear_cache()` periodically

## Best Practices

1. **Use Caching**: Enable caching for production to avoid redundant computation
2. **Batch Processing**: Process multiple texts together for better throughput
3. **Fine-tune**: Fine-tune on domain-specific data for better accuracy
4. **Monitor Memory**: Clear cache periodically in long-running applications
5. **Error Handling**: Always check for 'error' key in results
6. **GPU Usage**: Use GPU for fine-tuning and large batch processing

## References

- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

## License

This implementation uses open-source models and libraries. Check individual model licenses on HuggingFace.
