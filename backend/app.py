from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'polls.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'encoders.pkl')

class ElectionForecaster:
    def __init__(self):
        self.leaders = ['MK Stalin', 'Vijay', 'Edappadi K. Palaniswami', 'Annamalai', 'Seeman', 'Others']
        
    def get_mock_forecast(self):
        return {
            "cm_probabilities": {
                "MK Stalin": 0.45,
                "Vijay": 0.25,
                "Edappadi K. Palaniswami": 0.15,
                "Annamalai": 0.10,
                "Seeman": 0.03,
                "Others": 0.02
            },
            "vote_shares": {
                "MK Stalin": 42.5,
                "Vijay": 28.3,
                "Edappadi K. Palaniswami": 18.7,
                "Annamalai": 6.2,
                "Seeman": 2.8,
                "Others": 1.5
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
        result = forecaster.get_mock_forecast()
        return jsonify(result)
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
        return jsonify({"success": True, "message": "Mock training completed", "accuracy": 0.94})
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
