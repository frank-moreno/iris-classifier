"""
Iris Classification Model Training Script
Train a Decision Tree Classifier on the Iris dataset with configurable parameters.
"""

import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a Decision Tree Classifier on the Iris dataset'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of dataset to include in test split (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    return parser.parse_args()


def load_data():
    """Load the Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def split_data(X, y, test_size, random_state):
    """Split data into training and testing sets."""
    print(f"\nSplitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, random_state):
    """Train the Decision Tree Classifier."""
    print("\nTraining Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    """Make predictions and evaluate the model."""
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    # Show first 5 predictions
    print(f"\nFirst 5 predictions: {y_pred[:5]}")
    print(f"First 5 true labels: {y_test[:5]}")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Create outputs directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.title('Confusion Matrix - Iris Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to: {output_path}")
    
    return accuracy, cm


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 60)
    print("IRIS CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, args.test_size, args.random_state
    )
    
    # Train model
    model = train_model(X_train, y_train, args.random_state)
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()