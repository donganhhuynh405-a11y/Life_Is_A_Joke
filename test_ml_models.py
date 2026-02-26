#!/usr/bin/env python3
"""
Test script for ml_models.py to validate real ML implementation
"""
import numpy as np
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ml_models import PricePredictorLSTM, TransformerPredictor, HybridPredictor

def test_lstm_basic():
    """Test LSTM basic functionality"""
    print("\n" + "="*60)
    print("Testing LSTM Basic Functionality")
    print("="*60)
    
    try:
        # Create model
        lstm = PricePredictorLSTM(lookback=10, features=5, model_dir='test_models')
        print("✓ LSTM initialized")
        
        # Build model
        success = lstm.build_model()
        print(f"✓ LSTM model built: {success}")
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(200, 5)  # 200 samples, 5 features
        y = np.random.randint(0, 2, 200).astype(float)  # Binary targets
        
        print(f"✓ Generated test data: X={X.shape}, y={y.shape}")
        
        # Train model
        print("\nTraining LSTM (this may take a minute)...")
        metrics = lstm.train(X, y, epochs=5, batch_size=16)
        print(f"✓ Training completed: {metrics}")
        
        # Make predictions
        predictions = lstm.predict(X[:10])
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"  Sample predictions: {predictions[:3].flatten()}")
        
        # Save model
        saved = lstm.save_model('test_lstm')
        print(f"✓ Model saved: {saved}")
        
        # Load model
        lstm2 = PricePredictorLSTM(model_dir='test_models')
        loaded = lstm2.load_model('test_lstm')
        print(f"✓ Model loaded: {loaded}")
        
        # Verify loaded model works
        predictions2 = lstm2.predict(X[:10])
        print(f"✓ Loaded model predictions: {predictions2.shape}")
        
        # Evaluate
        eval_metrics = lstm.evaluate(X, y)
        print(f"✓ Evaluation metrics: {eval_metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transformer_basic():
    """Test Transformer basic functionality"""
    print("\n" + "="*60)
    print("Testing Transformer Basic Functionality")
    print("="*60)
    
    try:
        # Create model
        transformer = TransformerPredictor(
            max_len=50, d_model=8, heads=2, num_layers=1, 
            model_dir='test_models'
        )
        print("✓ Transformer initialized")
        
        # Build model
        success = transformer.build_model()
        print(f"✓ Transformer model built: {success}")
        
        # Generate synthetic sequence data
        np.random.seed(42)
        X = np.random.randn(100, 50, 8)  # 100 samples, 50 timesteps, 8 features
        y = np.random.randint(0, 3, 100)  # 3 classes: BUY, SELL, HOLD
        
        print(f"✓ Generated test data: X={X.shape}, y={y.shape}")
        
        # Train model
        print("\nTraining Transformer (this may take a minute)...")
        metrics = transformer.train(X, y, epochs=5, batch_size=16)
        print(f"✓ Training completed: {metrics}")
        
        # Make predictions
        test_features = X[0]
        prediction = transformer.predict(test_features)
        print(f"✓ Prediction: {prediction}")
        
        # Save model
        saved = transformer.save_model('test_transformer')
        print(f"✓ Model saved: {saved}")
        
        # Load model
        transformer2 = TransformerPredictor(model_dir='test_models')
        loaded = transformer2.load_model('test_transformer')
        print(f"✓ Model loaded: {loaded}")
        
        # Verify loaded model
        prediction2 = transformer2.predict(test_features)
        print(f"✓ Loaded model prediction: {prediction2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_ensemble():
    """Test Hybrid ensemble functionality"""
    print("\n" + "="*60)
    print("Testing Hybrid Ensemble")
    print("="*60)
    
    try:
        # Create ensemble
        hybrid = HybridPredictor(model_dir='test_models', lstm_weight=0.6, transformer_weight=0.4)
        print("✓ Hybrid predictor initialized")
        print(f"  Weights: LSTM={hybrid.lstm_weight:.2f}, Transformer={hybrid.transformer_weight:.2f}")
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(150, 5)
        y = np.random.randint(0, 2, 150).astype(float)
        
        print(f"✓ Generated test data: X={X.shape}, y={y.shape}")
        
        # Train ensemble
        print("\nTraining Hybrid Ensemble (this may take 2-3 minutes)...")
        metrics = hybrid.train(X, y, epochs=5, batch_size=16)
        print(f"✓ Training completed: {metrics}")
        
        # Make ensemble predictions
        predictions = hybrid.predict_ensemble(X[:5])
        print(f"✓ Ensemble prediction: {predictions}")
        
        # Save models
        saved = hybrid.save_models('test_hybrid')
        print(f"✓ Models saved: {saved}")
        
        # Load models
        hybrid2 = HybridPredictor(model_dir='test_models')
        loaded = hybrid2.load_models('test_hybrid')
        print(f"✓ Models loaded: {loaded}")
        
        # Verify loaded ensemble
        predictions2 = hybrid2.predict_ensemble(X[:5])
        print(f"✓ Loaded ensemble prediction: {predictions2}")
        
        # Evaluate
        eval_metrics = hybrid.evaluate(X, y)
        print(f"✓ Evaluation: {eval_metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hybrid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_random_values():
    """Verify that predictions are not random"""
    print("\n" + "="*60)
    print("Testing Non-Random Predictions")
    print("="*60)
    
    try:
        # Train a simple model
        lstm = PricePredictorLSTM(lookback=10, features=5, model_dir='test_models')
        lstm.build_model()
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100).astype(float)
        
        lstm.train(X, y, epochs=5, batch_size=16)
        
        # Make multiple predictions on same data
        pred1 = lstm.predict(X[:5])
        pred2 = lstm.predict(X[:5])
        
        # Predictions should be identical (deterministic)
        if np.allclose(pred1, pred2, rtol=1e-5):
            print("✓ Predictions are deterministic (not random)")
            print(f"  Prediction 1: {pred1.flatten()[:3]}")
            print(f"  Prediction 2: {pred2.flatten()[:3]}")
            return True
        else:
            print("✗ Predictions are not deterministic!")
            print(f"  Prediction 1: {pred1.flatten()[:3]}")
            print(f"  Prediction 2: {pred2.flatten()[:3]}")
            return False
            
    except Exception as e:
        print(f"✗ Determinism test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test model files"""
    print("\n" + "="*60)
    print("Cleaning up test files...")
    print("="*60)
    
    import shutil
    test_dir = Path('test_models')
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("✓ Test files cleaned up")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ML Models Production Implementation Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("LSTM Basic", test_lstm_basic()))
    results.append(("Transformer Basic", test_transformer_basic()))
    results.append(("Hybrid Ensemble", test_hybrid_ensemble()))
    results.append(("Non-Random Predictions", test_no_random_values()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Cleanup
    cleanup_test_files()
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
