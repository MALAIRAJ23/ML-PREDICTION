import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.db_path = '../data/polls.db'
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for evaluation"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM polls", conn)
        conn.close()
        
        if len(df) == 0:
            print("No data available for evaluation")
            return None, None, None, None
        
        # Preprocessing
        df = df.fillna({
            'sample_size': df['sample_size'].median(),
            'region': 'Tamil Nadu',
            'stalin_pct': 0,
            'vijay_pct': 0,
            'eps_pct': 0,
            'annamalai_pct': 0,
            'seeman_pct': 0,
            'others_pct': 0
        })
        
        # Time features
        df['fieldwork_date'] = pd.to_datetime(df['fieldwork_date'])
        df['days_to_election'] = (pd.to_datetime('2026-05-01') - df['fieldwork_date']).dt.days
        df['month'] = df['fieldwork_date'].dt.month
        df['quarter'] = df['fieldwork_date'].dt.quarter
        
        # Encode categorical variables
        le_org = LabelEncoder()
        le_region = LabelEncoder()
        
        df['organisation_encoded'] = le_org.fit_transform(df['organisation'].astype(str))
        df['region_encoded'] = le_region.fit_transform(df['region'].astype(str))
        
        # Create target variable
        pct_cols = ['stalin_pct', 'vijay_pct', 'eps_pct', 'annamalai_pct', 'seeman_pct', 'others_pct']
        df['cm_leader'] = df[pct_cols].idxmax(axis=1).map({
            'stalin_pct': 'MK Stalin',
            'vijay_pct': 'Vijay',
            'eps_pct': 'Edappadi K. Palaniswami',
            'annamalai_pct': 'Annamalai',
            'seeman_pct': 'Seeman',
            'others_pct': 'Others'
        })
        
        # Features
        feature_cols = ['sample_size', 'days_to_election', 'month', 'quarter', 
                       'organisation_encoded', 'region_encoded']
        X = df[feature_cols]
        y = df['cm_leader']
        
        return X, y, df, pct_cols
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("=== Tamil Nadu Election Forecasting Model Evaluation ===\n")
        
        X, y, df, pct_cols = self.load_and_preprocess_data()
        
        if X is None:
            return
        
        print(f"Dataset size: {len(X)} samples")
        print(f"Features: {list(X.columns)}")
        print(f"Target classes: {y.value_counts().to_dict()}\n")
        
        # Check if we have enough data and class diversity
        if len(y.unique()) < 2:
            print("Insufficient class diversity for evaluation")
            return
        
        if len(X) < 10:
            print("Insufficient data for proper evaluation")
            return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=600,
            class_weight='balanced',
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"=== Model Performance ===")
        print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"Cross-validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Classification report
        print(f"\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n=== Confusion Matrix ===")
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n=== Feature Importance ===")
        print(feature_importance)
        
        # Model confidence analysis
        confidence_scores = np.max(y_pred_proba, axis=1)
        print(f"\n=== Prediction Confidence ===")
        print(f"Mean confidence: {confidence_scores.mean():.3f}")
        print(f"Min confidence: {confidence_scores.min():.3f}")
        print(f"Max confidence: {confidence_scores.max():.3f}")
        
        # Accuracy by confidence threshold
        print(f"\n=== Accuracy by Confidence Threshold ===")
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_conf_mask = confidence_scores >= threshold
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = accuracy_score(
                    y_test[high_conf_mask], 
                    y_pred[high_conf_mask]
                )
                print(f"Confidence >= {threshold}: {high_conf_accuracy:.3f} ({high_conf_mask.sum()} samples)")
        
        # Performance assessment
        print(f"\n=== Performance Assessment ===")
        if accuracy >= 0.8:
            print("✅ Excellent performance (≥80%)")
        elif accuracy >= 0.6:
            print("✅ Good performance (60-80% - Target achieved)")
        elif accuracy >= 0.4:
            print("⚠️  Fair performance (40-60%)")
        else:
            print("❌ Poor performance (<40%)")
        
        return {
            'accuracy': accuracy,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

def main():
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model()
    
    if results:
        print(f"\n=== Summary ===")
        print(f"Model meets target accuracy: {'Yes' if results['accuracy'] >= 0.6 else 'No'}")
        print(f"Ready for production: {'Yes' if results['accuracy'] >= 0.6 else 'No'}")

if __name__ == '__main__':
    main()