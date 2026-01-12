import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
import joblib
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'polls.db')
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for training"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM polls", conn)
        conn.close()
        
        if len(df) == 0:
            raise ValueError("No data available for training")
        
        # Fill missing values
        df = df.fillna({
            'sample_size': df['sample_size'].median(),
            'region': 'Tamil Nadu',
            'stalin_pct': 0, 'vijay_pct': 0, 'eps_pct': 0,
            'annamalai_pct': 0, 'seeman_pct': 0, 'others_pct': 0
        })
        
        # Time features
        df['fieldwork_date'] = pd.to_datetime(df['fieldwork_date'])
        df['days_to_election'] = (pd.to_datetime('2026-05-01') - df['fieldwork_date']).dt.days
        df['month'] = df['fieldwork_date'].dt.month
        df['quarter'] = df['fieldwork_date'].dt.quarter
        
        # Encode categorical variables
        self.le_org = LabelEncoder()
        self.le_region = LabelEncoder()
        
        df['organisation_encoded'] = self.le_org.fit_transform(df['organisation'].astype(str))
        df['region_encoded'] = self.le_region.fit_transform(df['region'].astype(str))
        
        # Create target variable (winner)
        pct_cols = ['stalin_pct', 'vijay_pct', 'eps_pct', 'annamalai_pct', 'seeman_pct', 'others_pct']
        df['winner'] = df[pct_cols].idxmax(axis=1).map({
            'stalin_pct': 'MK Stalin',
            'vijay_pct': 'Vijay',
            'eps_pct': 'Edappadi K. Palaniswami',
            'annamalai_pct': 'Annamalai',
            'seeman_pct': 'Seeman',
            'others_pct': 'Others'
        })
        
        # Features for prediction
        feature_cols = ['sample_size', 'days_to_election', 'month', 'quarter', 
                       'organisation_encoded', 'region_encoded']
        X = df[feature_cols]
        y = df['winner']
        
        # Also prepare vote share data
        vote_shares = df[pct_cols]
        
        return X, y, vote_shares, df
    
    def train_and_save_models(self):
        """Train and save ML models"""
        print("Loading and preprocessing data...")
        X, y, vote_shares, df = self.load_and_preprocess_data()
        
        print(f"Training on {len(X)} samples with {len(y.unique())} classes")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train winner prediction model
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            max_depth=10
        )
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.3f}")
        
        # Save models
        joblib.dump(self.model, os.path.join(self.model_dir, 'rf_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump({
            'le_org': self.le_org,
            'le_region': self.le_region
        }, os.path.join(self.model_dir, 'encoders.pkl'))
        
        # Calculate and save vote share averages for prediction
        vote_share_means = vote_shares.mean().to_dict()
        joblib.dump(vote_share_means, os.path.join(self.model_dir, 'vote_shares.pkl'))
        
        print("Models saved successfully!")
        return accuracy

if __name__ == '__main__':
    trainer = ModelTrainer()
    accuracy = trainer.train_and_save_models()
    print(f"Training completed with accuracy: {accuracy:.3f}")