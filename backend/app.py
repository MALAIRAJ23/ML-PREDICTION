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
        # Return mock poll data
        mock_polls = [
            {
                "id": 1,
                "organisation": "C-Voter",
                "fieldwork_date": "2025-12-15",
                "sample_size": 2500,
                "region": "Tamil Nadu",
                "stalin_pct": 42.5,
                "vijay_pct": 28.3,
                "eps_pct": 18.7,
                "annamalai_pct": 6.2,
                "seeman_pct": 2.8,
                "others_pct": 1.5
            },
            {
                "id": 2,
                "organisation": "ABP-CVoter",
                "fieldwork_date": "2025-12-10",
                "sample_size": 1800,
                "region": "Tamil Nadu",
                "stalin_pct": 44.2,
                "vijay_pct": 26.8,
                "eps_pct": 19.5,
                "annamalai_pct": 5.8,
                "seeman_pct": 2.2,
                "others_pct": 1.5
            }
        ]
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
