"""
Model evaluation script
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import FEATURES_DIR, SAVED_MODELS_DIR
from models.user_based_cf import UserBasedCF
from models.item_based_cf import ItemBasedCF
from models.neural_cf import NeuralCF
from models.hybrid_model import HybridModel

def load_test_data():
    """Load test data"""
    print("Loading test data...")
    test_ratings = pd.read_csv(FEATURES_DIR / "test_ratings.csv")
    return test_ratings

def load_models():
    """Load trained models"""
    print("Loading trained models...")
    
    user_based = UserBasedCF.load(SAVED_MODELS_DIR / "user_based_cf.pkl")
    item_based = ItemBasedCF.load(SAVED_MODELS_DIR / "item_based_cf.pkl")
    neural_cf = NeuralCF.load(SAVED_MODELS_DIR / "neural_cf.pkl")
    hybrid = HybridModel.load(
        SAVED_MODELS_DIR / "hybrid_model.pkl",
        user_based,
        item_based,
        neural_cf
    )
    
    return {
        'user_based': user_based,
        'item_based': item_based,
        'neural_cf': neural_cf,
        'hybrid': hybrid
    }

def evaluate_rmse_mae(model, test_ratings, model_name):
    """Evaluate RMSE and MAE"""
    print(f"\nEvaluating {model_name}...")
    
    predictions = []
    actuals = []
    
    # Sample for efficiency (evaluate on subset if dataset is very large)
    sample_size = min(10000, len(test_ratings))
    test_sample = test_ratings.sample(n=sample_size, random_state=42)
    
    for _, row in test_sample.iterrows():
        pred = model.predict(row['userId'], row['movieId'])
        predictions.append(pred)
        actuals.append(row['rating'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    return rmse, mae

def evaluate_precision_recall_at_k(model, test_ratings, k=10, model_name="Model"):
    """Evaluate Precision@K and Recall@K"""
    print(f"Evaluating Precision@{k} and Recall@{k} for {model_name}...")
    
    # Sample users
    users = test_ratings['userId'].unique()
    sample_users = np.random.choice(users, size=min(100, len(users)), replace=False)
    
    precisions = []
    recalls = []
    
    for user_id in sample_users:
        # Get user's test items
        user_test = test_ratings[test_ratings['userId'] == user_id]
        
        # Only consider highly rated items (rating >= 4.0) as relevant
        relevant_items = set(user_test[user_test['rating'] >= 4.0]['movieId'])
        
        if len(relevant_items) == 0:
            continue
        
        # Get recommendations
        try:
            recommendations = model.recommend(user_id, k=k, exclude_rated=True)
            recommended_items = set([movie_id for movie_id, _ in recommendations])
            
            # Calculate metrics
            hits = len(recommended_items & relevant_items)
            precision = hits / k if k > 0 else 0
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        except:
            continue
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    return avg_precision, avg_recall

def evaluate_model(model, test_ratings, model_name, k=10):
    """Evaluate all metrics for a model"""
    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name.upper()}")
    print("=" * 60)
    
    # RMSE and MAE
    rmse, mae = evaluate_rmse_mae(model, test_ratings, model_name)
    
    # Precision and Recall at K
    precision, recall = evaluate_precision_recall_at_k(model, test_ratings, k, model_name)
    
    metrics = {
        'model': model_name,
        'rmse': float(rmse),
        'mae': float(mae),
        f'precision@{k}': float(precision),
        f'recall@{k}': float(recall)
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Precision@{k}: {precision:.4f}")
    print(f"  Recall@{k}: {recall:.4f}")
    
    return metrics

def save_metrics(all_metrics):
    """Save metrics to JSON file"""
    output_file = SAVED_MODELS_DIR / "evaluation_metrics.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_file}")

def print_comparison(all_metrics):
    """Print comparison table"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'Precision@10':<15} {'Recall@10':<15}")
    print("-" * 75)
    
    for metrics in all_metrics:
        print(f"{metrics['model']:<20} "
              f"{metrics['rmse']:<10.4f} "
              f"{metrics['mae']:<10.4f} "
              f"{metrics['precision@10']:<15.4f} "
              f"{metrics['recall@10']:<15.4f}")
    
    # Find best models
    print("\nBest Models:")
    best_rmse = min(all_metrics, key=lambda x: x['rmse'])
    best_mae = min(all_metrics, key=lambda x: x['mae'])
    best_precision = max(all_metrics, key=lambda x: x['precision@10'])
    best_recall = max(all_metrics, key=lambda x: x['recall@10'])
    
    print(f"  Best RMSE: {best_rmse['model']} ({best_rmse['rmse']:.4f})")
    print(f"  Best MAE: {best_mae['model']} ({best_mae['mae']:.4f})")
    print(f"  Best Precision@10: {best_precision['model']} ({best_precision['precision@10']:.4f})")
    print(f"  Best Recall@10: {best_recall['model']} ({best_recall['recall@10']:.4f})")

def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load data and models
    test_ratings = load_test_data()
    models = load_models()
    
    # Evaluate each model
    all_metrics = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, test_ratings, model_name, k=10)
        all_metrics.append(metrics)
    
    # Save metrics
    save_metrics(all_metrics)
    
    # Print comparison
    print_comparison(all_metrics)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
