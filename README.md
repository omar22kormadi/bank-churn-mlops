# Bank Churn Prediction API

A production-ready machine learning application that predicts customer churn using a trained scikit-learn model, served through a FastAPI REST API with Docker support.

## 🎯 Overview

This project builds and deploys a binary classification model to predict whether a bank customer will leave (churn) based on their financial and demographic characteristics. The system includes:

- **ML Pipeline**: Model training and evaluation with MLflow tracking
- **REST API**: FastAPI endpoint for real-time and batch predictions
- **Containerization**: Docker support for easy deployment
- **CORS Support**: Cross-origin requests enabled for web integration

## 📊 Project Structure

```
ml-deploy/
├── app/                          # FastAPI application
│   ├── __init__.py
│   ├── main.py                  # API endpoints and startup logic
│   ├── models.py                # Pydantic request/response schemas
│   └── utils.py                 # Helper utilities
├── model/                        # Trained model storage
│   └── churn_model.pkl          # Serialized model file
├── data/
│   └── bank_churn.csv           # Training dataset (10,000+ records)
├── mlruns/                       # MLflow experiment tracking
│   └── models/
│       └── bank-churn-classifier/
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── train_model.py               # Model training script
├── generate_data.py             # Synthetic data generation
└── README.md                     # This file
```

## 📋 Features

- ✅ **Real-time Predictions**: Single customer churn probability prediction
- ✅ **Batch Processing**: Predict for multiple customers in one request
- ✅ **Risk Classification**: Automatic risk level assignment (Low/Medium/High)
- ✅ **Model Versioning**: MLflow integration for experiment tracking
- ✅ **Health Checks**: API status and model availability monitoring
- ✅ **Comprehensive Logging**: Request/response logging for debugging
- ✅ **Interactive Documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`
- ✅ **Input Validation**: Pydantic schema validation with detailed error messages
- ✅ **CORS Enabled**: Safe cross-origin requests from web applications

## 🔧 Installation

### Prerequisites

- Python 3.9+
- pip or conda
- Docker (optional, for containerization)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-deploy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file**
   ```bash
   # Ensure model/churn_model.pkl exists
   ls model/churn_model.pkl
   ```

## 🚀 Usage

### Training the Model

Train a new model using the provided dataset:

```bash
python train_model.py
```

This script:
- Loads `data/bank_churn.csv`
- Preprocesses features (scaling, encoding)
- Trains a Random Forest classifier
- Logs metrics to MLflow
- Saves the model to `model/churn_model.pkl`

### Generating Synthetic Data

Generate additional training data:

```bash
python generate_data.py
```

### Running the API

**Locally:**
```bash
python app/main.py
```

The API will start at `http://localhost:8000`

**Using Uvicorn directly:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

**Build the image:**
```bash
docker build -t bank-churn-api:latest .
```

**Run the container:**
```bash
docker run -p 8000:8000 bank-churn-api:latest
```

**With custom model path:**
```bash
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/model/custom_model.pkl \
  bank-churn-api:latest
```

## 📡 API Endpoints

All endpoints are documented interactively at `http://localhost:8000/docs`

### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Single Prediction
```
POST /predict
```

**Request Body:**
```json
{
  "CreditScore": 650,
  "Age": 35,
  "Tenure": 5,
  "Balance": 50000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 75000,
  "Geography_Germany": 0,
  "Geography_Spain": 1
}
```

**Response:**
```json
{
  "churn_probability": 0.2345,
  "prediction": 0,
  "risk_level": "Low"
}
```

**Field Descriptions:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| CreditScore | integer | 300-850 | Customer credit score |
| Age | integer | 18-100 | Customer age in years |
| Tenure | integer | 0-10 | Years as customer |
| Balance | float | ≥0 | Account balance in currency units |
| NumOfProducts | integer | 1-4 | Number of bank products used |
| HasCrCard | integer | 0-1 | 1 if has credit card, 0 otherwise |
| IsActiveMember | integer | 0-1 | 1 if actively using account, 0 otherwise |
| EstimatedSalary | float | ≥0 | Estimated annual salary |
| Geography_Germany | integer | 0-1 | 1 if customer from Germany, 0 otherwise |
| Geography_Spain | integer | 0-1 | 1 if customer from Spain, 0 otherwise |

### 3. Batch Predictions
```
POST /predict/batch
```

**Request Body:**
```json
[
  {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
  },
  {
    "CreditScore": 720,
    "Age": 45,
    "Tenure": 8,
    "Balance": 75000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 95000,
    "Geography_Germany": 1,
    "Geography_Spain": 0
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "churn_probability": 0.2345,
      "prediction": 0
    },
    {
      "churn_probability": 0.1234,
      "prediction": 0
    }
  ],
  "count": 2
}
```

### 4. Root Endpoint
```
GET /
```

**Response:**
```json
{
  "message": "Bank Churn Prediction API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

## 📊 Model Information

### Model Type
- **Algorithm**: Random Forest Classifier
- **Features**: 10 engineered features
- **Output**: Binary classification (0: customer stays, 1: customer churns)

### Performance Metrics
Metrics are logged during training and stored in MLflow:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Percentage of predicted churners who actually churn
- **Recall**: Percentage of actual churners identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

### Risk Level Classification
- **Low**: Churn probability < 0.3
- **Medium**: Churn probability 0.3 - 0.7
- **High**: Churn probability > 0.7

## 📦 Dependencies

Core dependencies (see `requirements.txt` for versions):

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Pydantic**: Data validation and settings management
- **scikit-learn**: Machine learning library
- **joblib**: Model serialization
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **MLflow**: Experiment tracking and model registry
- **python-dotenv**: Environment variable management

## 🔐 Environment Variables

Configure these optional environment variables:

```bash
# Path to the trained model (default: model/churn_model.pkl)
MODEL_PATH=model/churn_model.pkl

# API host (default: 0.0.0.0)
API_HOST=0.0.0.0

# API port (default: 8000)
API_PORT=8000

# Log level (default: INFO)
LOG_LEVEL=INFO
```

## 🧪 Testing

### Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
  }'

# Batch predictions
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[...]'
```

### Test with Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
}

response = requests.post(url, json=data)
print(response.json())
```

## 📈 MLflow Integration

Track experiments and manage models with MLflow:

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

Features:
- View all training runs and metrics
- Compare model performance across experiments
- Register best models to model registry
- Track hyperparameters and artifacts

Model Registry path: `mlruns/models/bank-churn-classifier/`

## 🐳 Docker Details

### Dockerfile Features
- Multi-stage build for optimization
- Python 3.9-slim base image (minimal footprint)
- Non-root user for security (optional hardening)
- Health check endpoint support

### Image Information
- **Base**: `python:3.9-slim`
- **Working Directory**: `/app`
- **Exposed Port**: 8000
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Docker Compose (Optional)

Create `docker-compose.yml` for local development:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=model/churn_model.pkl
    volumes:
      - ./model:/app/model
    restart: unless-stopped
```

Start with: `docker-compose up -d`

## 🚨 Error Handling

The API returns standard HTTP status codes:

| Code | Scenario |
|------|----------|
| 200 | Successful prediction |
| 400 | Invalid input (bad request) |
| 422 | Validation error (invalid schema) |
| 500 | Server error (prediction failed) |
| 503 | Model not loaded (service unavailable) |

**Example Error Response:**
```json
{
  "detail": "Model not available"
}
```

## 📝 Logging

Logs are output to console with INFO level by default:

```
INFO:uvicorn.access:127.0.0.1:12345 - "POST /predict HTTP/1.1" 200 OK
INFO:app.main:Prediction made: proba=0.2345, prediction=0, risk=Low
```

## 🔄 Deployment Workflows

### Development
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API with auto-reload
uvicorn app.main:app --reload

# Access docs
open http://localhost:8000/docs
```

### Production
```bash
# Build Docker image
docker build -t bank-churn-api:v1.0 .

# Push to registry
docker push myregistry.azurecr.io/bank-churn-api:v1.0

# Deploy to cloud (example: Azure Container Instances)
az container create \
  --resource-group mygroup \
  --name bank-churn-api \
  --image myregistry.azurecr.io/bank-churn-api:v1.0 \
  --ports 8000 \
  --cpu 1 --memory 1
```

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "Add feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Open a Pull Request

## 📄 License

This project is provided as-is for educational and commercial use.

## 📧 Support

For issues, questions, or feature requests, please open an issue on the repository.

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [REST API Design Guidelines](https://restfulapi.net/)

## 📌 Version History

- **v1.0.0** (Jan 2026): Initial release with prediction and batch endpoints

---

**Last Updated**: January 2026
