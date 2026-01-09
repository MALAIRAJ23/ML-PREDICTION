from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sqlite3
import pickle
import os
from datetime import datetime, timedelta
import hashlib

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'polls.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'encoders.pkl')

class ElectionForecaster:
    def __init__(self):
        self.cm_model = None
        self.vote_share_model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.leaders = ['MK Stalin', 'Vijay', 'Edappadi K. Palaniswami', 'Annamalai', 'Seeman', 'Others']
        
    def load_data(self):
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query("""
            SELECT * FROM polls 
            WHERE created_at >= date('now', '-6 months')
            ORDER BY fieldwork_date DESC
        """, conn)
        conn.close()
        return df
    
    def preprocess_data(self, df):
        # Handle missing values
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
        
        # Normalize percentages
        pct_cols = ['stalin_pct', 'vijay_pct', 'eps_pct', 'annamalai_pct', 'seeman_pct', 'others_pct']
        for col in pct_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = np.clip(df[col], 0, 100)
        
        # Time-based features
        df['fieldwork_date'] = pd.to_datetime(df['fieldwork_date'])
        df['days_to_election'] = (pd.to_datetime('2026-05-01') - df['fieldwork_date']).dt.days
        df['month'] = df['fieldwork_date'].dt.month
        df['quarter'] = df['fieldwork_date'].dt.quarter
        
        # Encode categorical variables
        categorical_cols = ['organisation', 'region']
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        # Create target variable (CM preference leader)
        df['cm_leader'] = df[pct_cols].idxmax(axis=1).map({
            'stalin_pct': 'MK Stalin',
            'vijay_pct': 'Vijay', 
            'eps_pct': 'Edappadi K. Palaniswami',
            'annamalai_pct': 'Annamalai',
            'seeman_pct': 'Seeman',
            'others_pct': 'Others'
        })
        
        return df
    
    def train_models(self):
        df = self.load_data()
        if len(df) < 10:
            return {"error": "Insufficient data for training"}
        
        df = self.preprocess_data(df)
        
        # Features for training
        feature_cols = ['sample_size', 'days_to_election', 'month', 'quarter', 
                       'organisation_encoded', 'region_encoded']
        X = df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # CM preference classification
        y_cm = df['cm_leader']
        if len(y_cm.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cm, test_size=0.2, random_state=42)
            
            self.cm_model = RandomForestClassifier(
                n_estimators=600,
                class_weight='balanced',
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.cm_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.cm_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Vote share regression
            vote_cols = ['stalin_pct', 'vijay_pct', 'eps_pct', 'annamalai_pct', 'seeman_pct', 'others_pct']
            y_votes = df[vote_cols]
            
            X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_scaled, y_votes, test_size=0.2, random_state=42)
            
            self.vote_share_model = RandomForestRegressor(
                n_estimators=600,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.vote_share_model.fit(X_train_v, y_train_v)
            
            # Save models
            self.save_models()
            
            return {
                "success": True,
                "accuracy": accuracy,
                "samples": len(df),
                "features": feature_cols
            }
        
        return {"error": "Insufficient class diversity"}
    
    def save_models(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'cm_model': self.cm_model,
                'vote_share_model': self.vote_share_model
            }, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(ENCODERS_PATH, 'wb') as f:
            pickle.dump(self.encoders, f)
    
    def load_models(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                models = pickle.load(f)
                self.cm_model = models['cm_model']
                self.vote_share_model = models['vote_share_model']
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(ENCODERS_PATH, 'rb') as f:
                self.encoders = pickle.load(f)
            return True
        except:
            return False
    
    def predict(self, sample_data=None):
        if not self.cm_model:
            if not self.load_models():
                return {"error": "No trained model available"}
        
        # Use latest poll data for prediction
        df = self.load_data()
        if len(df) == 0:
            return {"error": "No data available"}
        
        df = self.preprocess_data(df)
        latest_data = df.iloc[0:1]
        
        feature_cols = ['sample_size', 'days_to_election', 'month', 'quarter', 
                       'organisation_encoded', 'region_encoded']
        X = latest_data[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # CM preference prediction
        cm_proba = self.cm_model.predict_proba(X_scaled)[0]
        cm_classes = self.cm_model.classes_
        
        # Vote share prediction
        vote_pred = self.vote_share_model.predict(X_scaled)[0]
        
        return {
            "cm_probabilities": dict(zip(cm_classes, cm_proba)),
            "vote_shares": {
                "MK Stalin": max(0, min(100, vote_pred[0])),
                "Vijay": max(0, min(100, vote_pred[1])),
                "Edappadi K. Palaniswami": max(0, min(100, vote_pred[2])),
                "Annamalai": max(0, min(100, vote_pred[3])),
                "Seeman": max(0, min(100, vote_pred[4])),
                "Others": max(0, min(100, vote_pred[5]))
            },
            "prediction_date": datetime.now().isoformat()
        }

forecaster = ElectionForecaster()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Tamil Nadu 2026 Election Forecasting API",
        "status": "running",
        "endpoints": ["/api/forecast", "/api/polls", "/api/train", "/api/add_poll"]
    })

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        result = forecaster.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/polls', methods=['GET'])
def get_polls():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        df = pd.read_sql_query("""
            SELECT * FROM polls 
            ORDER BY fieldwork_date DESC 
            LIMIT 50
        """, conn)
        conn.close()
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        # Simple authentication
        auth = request.json.get('password')
        if auth != 'admin123':
            return jsonify({"error": "Unauthorized"}), 401
        
        result = forecaster.train_models()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_poll', methods=['POST'])
def add_poll():
    try:
        data = request.json
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Insert poll data
        conn.execute("""
            INSERT INTO polls (organisation, fieldwork_date, sample_size, region,
                             stalin_pct, vijay_pct, eps_pct, annamalai_pct, seeman_pct, others_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('organisation'),
            data.get('fieldwork_date'),
            data.get('sample_size'),
            data.get('region', 'Tamil Nadu'),
            data.get('stalin_pct', 0),
            data.get('vijay_pct', 0),
            data.get('eps_pct', 0),
            data.get('annamalai_pct', 0),
            data.get('seeman_pct', 0),
            data.get('others_pct', 0)
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "tn-election-backend"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
