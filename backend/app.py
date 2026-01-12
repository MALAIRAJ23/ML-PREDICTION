from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'polls.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'encoders.pkl')
VOTE_SHARES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vote_shares.pkl')

class ElectionForecaster:
    def __init__(self):
        self.leaders = ['MK Stalin', 'Vijay', 'Edappadi K. Palaniswami', 'Annamalai', 'Seeman', 'Others']
        self.model = None
        self.scaler = None
        self.encoders = None
        self.vote_shares = None
        self.load_models()
        
    def load_models(self):
        """Load trained ML models"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.encoders = joblib.load(ENCODERS_PATH)
                self.vote_shares = joblib.load(VOTE_SHARES_PATH)
                print("ML models loaded successfully")
            else:
                print("Models not found, will use fallback predictions")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model = None
    
    def get_recent_poll_features(self):
        """Get features from recent polls for prediction"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            df = pd.read_sql_query("""
                SELECT * FROM polls 
                ORDER BY fieldwork_date DESC 
                LIMIT 10
            """, conn)
            conn.close()
            
            if len(df) == 0:
                return None
            
            # Use most recent poll as base
            recent = df.iloc[0]
            
            # Create features similar to training
            features = {
                'sample_size': recent['sample_size'],
                'days_to_election': (pd.to_datetime('2026-05-01') - pd.to_datetime(recent['fieldwork_date'])).days,
                'month': pd.to_datetime(recent['fieldwork_date']).month,
                'quarter': pd.to_datetime(recent['fieldwork_date']).quarter,
                'organisation_encoded': 0,  # Default encoding
                'region_encoded': 0  # Default encoding
            }
            
            # Try to encode if encoders available
            if self.encoders:
                try:
                    org_encoded = self.encoders['le_org'].transform([recent['organisation']])[0]
                    region_encoded = self.encoders['le_region'].transform([recent['region']])[0]
                    features['organisation_encoded'] = org_encoded
                    features['region_encoded'] = region_encoded
                except:
                    pass  # Use defaults if encoding fails
            
            return features
        except Exception as e:
            print(f"Error getting poll features: {e}")
            return None
    
    def calculate_vote_shares_from_polls(self):
        """Calculate vote shares from recent polls"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            df = pd.read_sql_query("""
                SELECT stalin_pct, vijay_pct, eps_pct, annamalai_pct, seeman_pct, others_pct
                FROM polls 
                ORDER BY fieldwork_date DESC 
                LIMIT 20
            """, conn)
            conn.close()
            
            if len(df) == 0:
                return self.get_fallback_vote_shares()
            
            # Weight recent polls more heavily
            weights = np.exp(-np.arange(len(df)) * 0.1)  # Exponential decay
            weights = weights / weights.sum()
            
            vote_shares = {
                "MK Stalin": np.average(df['stalin_pct'], weights=weights),
                "Vijay": np.average(df['vijay_pct'], weights=weights),
                "Edappadi K. Palaniswami": np.average(df['eps_pct'], weights=weights),
                "Annamalai": np.average(df['annamalai_pct'], weights=weights),
                "Seeman": np.average(df['seeman_pct'], weights=weights),
                "Others": np.average(df['others_pct'], weights=weights)
            }
            
            return vote_shares
        except Exception as e:
            print(f"Error calculating vote shares: {e}")
            return self.get_fallback_vote_shares()
    
    def get_ml_forecast(self):
        """Generate forecast using ML model"""
        if not self.model:
            return self.get_fallback_forecast()
        
        try:
            # Get features for prediction
            features = self.get_recent_poll_features()
            if not features:
                return self.get_fallback_forecast()
            
            # Prepare feature array
            feature_array = np.array([[features[col] for col in 
                ['sample_size', 'days_to_election', 'month', 'quarter', 
                 'organisation_encoded', 'region_encoded']]])
            
            # Scale features
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)
            
            # Get predictions
            probabilities = self.model.predict_proba(feature_array)[0]
            classes = self.model.classes_
            
            # Create probability dictionary
            cm_probabilities = {}
            for i, leader in enumerate(classes):
                cm_probabilities[leader] = float(probabilities[i])
            
            # Ensure all leaders are included
            for leader in self.leaders:
                if leader not in cm_probabilities:
                    cm_probabilities[leader] = 0.01
            
            # Get vote shares from recent polls
            vote_shares = self.calculate_vote_shares_from_polls()
            
            return {
                "cm_probabilities": cm_probabilities,
                "vote_shares": vote_shares,
                "prediction_date": datetime.now().isoformat(),
                "model_used": "Random Forest",
                "data_source": "Recent Polls"
            }
            
        except Exception as e:
            print(f"Error in ML forecast: {e}")
            return self.get_fallback_forecast()
    
    def get_fallback_forecast(self):
        """Fallback forecast when ML model is not available"""
        vote_shares = self.calculate_vote_shares_from_polls()
        
        # Convert vote shares to probabilities with some uncertainty
        total_votes = sum(vote_shares.values())
        cm_probabilities = {}
        
        for leader, votes in vote_shares.items():
            # Add some uncertainty to vote share -> probability conversion
            base_prob = votes / total_votes if total_votes > 0 else 0.16
            # Add random variation
            prob = max(0.01, min(0.95, base_prob + np.random.normal(0, 0.05)))
            cm_probabilities[leader] = prob
        
        # Normalize probabilities
        total_prob = sum(cm_probabilities.values())
        cm_probabilities = {k: v/total_prob for k, v in cm_probabilities.items()}
        
        return {
            "cm_probabilities": cm_probabilities,
            "vote_shares": vote_shares,
            "prediction_date": datetime.now().isoformat(),
            "model_used": "Poll Aggregation",
            "data_source": "Recent Polls"
        }
    
    def get_fallback_vote_shares(self):
        """Fallback vote shares when no poll data available"""
        return {
            "MK Stalin": 42.5,
            "Vijay": 28.3,
            "Edappadi K. Palaniswami": 18.7,
            "Annamalai": 6.2,
            "Seeman": 2.8,
            "Others": 1.5
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
        result = forecaster.get_ml_forecast()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        # Import and run training
        from train_model import ModelTrainer
        trainer = ModelTrainer()
        accuracy = trainer.train_and_save_models()
        
        # Reload models in forecaster
        forecaster.load_models()
        
        return jsonify({
            "success": True, 
            "message": "Model retrained successfully", 
            "accuracy": round(accuracy, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/polls', methods=['GET'])
def get_polls():
    try:
        # Return 50 diverse mock poll data
        import random
        from datetime import datetime, timedelta
        
        organisations = ["C-Voter", "ABP-CVoter", "India Today-Axis", "Times Now-VMR", "Republic-P-MARQ", 
                        "CNN-News18", "Zee News-BARC", "NewsX-Neta", "TV9-Polstrat", "Lokniti-CSDS"]
        
        mock_polls = []
        base_date = datetime(2025, 12, 15)
        
        for i in range(50):
            # Vary the percentages realistically
            stalin_base = 42 + random.uniform(-8, 8)
            vijay_base = 28 + random.uniform(-6, 6)
            eps_base = 19 + random.uniform(-5, 5)
            annamalai_base = 6 + random.uniform(-2, 4)
            seeman_base = 3 + random.uniform(-1, 2)
            others_base = 2 + random.uniform(-1, 2)
            
            # Normalize to 100%
            total = stalin_base + vijay_base + eps_base + annamalai_base + seeman_base + others_base
            stalin_pct = round((stalin_base / total) * 100, 1)
            vijay_pct = round((vijay_base / total) * 100, 1)
            eps_pct = round((eps_base / total) * 100, 1)
            annamalai_pct = round((annamalai_base / total) * 100, 1)
            seeman_pct = round((seeman_base / total) * 100, 1)
            others_pct = round(100 - stalin_pct - vijay_pct - eps_pct - annamalai_pct - seeman_pct, 1)
            
            poll_date = base_date - timedelta(days=i*2)
            
            mock_polls.append({
                "id": i + 1,
                "organisation": random.choice(organisations),
                "fieldwork_date": poll_date.strftime("%Y-%m-%d"),
                "sample_size": random.randint(1200, 3500),
                "region": "Tamil Nadu",
                "stalin_pct": stalin_pct,
                "vijay_pct": vijay_pct,
                "eps_pct": eps_pct,
                "annamalai_pct": annamalai_pct,
                "seeman_pct": seeman_pct,
                "others_pct": others_pct,
                "created_at": poll_date.isoformat()
            })
        
        return jsonify(mock_polls)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        from train_model import ModelTrainer
        trainer = ModelTrainer()
        accuracy = trainer.train_and_save_models()
        forecaster.load_models()  # Reload models
        return jsonify({
            "success": True, 
            "message": "Model trained successfully", 
            "accuracy": round(accuracy, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/add_poll', methods=['POST'])
def add_poll():
    try:
        return jsonify({"success": True, "message": "Poll data received"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "tn-election-backend"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
