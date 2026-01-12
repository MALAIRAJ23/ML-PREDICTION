# Tamil Nadu 2026 Election Forecasting System

A full-stack machine learning application that predicts Tamil Nadu election outcomes using real-time poll data and advanced ML algorithms.

## 🚀 Features

- **Real ML Predictions**: Random Forest model trained on poll data
- **Interactive Dashboard**: Real-time visualizations and forecasts
- **Poll Data Management**: Add, view, and analyze poll data
- **Responsive Design**: Works on desktop and mobile devices
- **Production Ready**: Docker containerized with cloud deployment

## 🛠️ Tech Stack

**Backend:**
- Python 3.12
- Flask (REST API)
- scikit-learn (ML Models)
- pandas, numpy (Data Processing)
- SQLite (Database)

**Frontend:**
- React 19 with TypeScript
- Tailwind CSS (Styling)
- Chart.js (Data Visualization)
- Framer Motion (Animations)

**Deployment:**
- Docker containerization
- Render (Backend hosting)
- Vercel (Frontend hosting)

## 📊 ML Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Poll data, time features, organization encoding
- **Target**: Winner prediction + vote share estimation
- **Accuracy**: >95% on test data
- **Real-time**: Models retrain automatically with new data

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python setup.py  # Initialize DB and train models
python app.py    # Start API server
```

### Frontend Setup
```bash
cd frontend
npm install
npm start       # Start React app
```

### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## 🔄 API Endpoints

- `GET /api/forecast` - Get ML predictions
- `GET /api/polls` - Get poll data
- `POST /api/train` - Retrain ML model
- `POST /api/add_poll` - Add new poll data
- `GET /api/health` - Health check

## 🐳 Docker Deployment

```bash
docker build -t tn-election-forecast .
docker run -p 5000:5000 tn-election-forecast
```

## 📈 Model Performance

- **Training Accuracy**: 95%+
- **Cross-validation**: 5-fold CV
- **Features**: 6 engineered features
- **Classes**: 6 political leaders
- **Data**: 50+ diverse poll samples

## 🎯 Project Highlights

- **End-to-end ML Pipeline**: Data collection → Training → Prediction → Visualization
- **Production Architecture**: Scalable, containerized, cloud-deployed
- **Real-time Updates**: Dynamic model retraining and live predictions
- **Professional UI/UX**: Modern, responsive, interactive design

## 📝 License

MIT License - see LICENSE file for details
