from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from flask_cors import CORS
import logging
from functools import wraps
import time
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import boto3
import json

load_dotenv()
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config.update(
    PREDICTION_API_URL=os.getenv('PREDICTION_API_URL'),
    API_KEY=os.getenv('API_GATEWAY_KEY'),
    REQUEST_TIMEOUT=10,
    AWS_REGION=os.getenv('AWS_REGION', 'ap-southeast-1'),
    DYNAMODB_TABLE=os.getenv('DYNAMODB_TABLE', 'ProductEmbeddings')
)

# Initialize AWS clients with error handling
try:
    boto3.setup_default_session(
        region_name=app.config['AWS_REGION'],
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN')
    )
    dynamodb = boto3.resource('dynamodb')
    
    # Verify table exists
    table = dynamodb.Table(app.config['DYNAMODB_TABLE'])
    table.load()
    logger.info(f"Successfully connected to DynamoDB table: {app.config['DYNAMODB_TABLE']}")
    
except Exception as e:
    logger.error(f"AWS initialization failed: {str(e)}")
    dynamodb = None
    table = None

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting request: {request.method} {request.path}")
        
        try:
            response = f(*args, **kwargs)
            duration = round((time.time() - start_time) * 1000, 2)
            
            if isinstance(response, tuple):
                response_obj = response[0]
                status_code = response[1] if len(response) > 1 else 200
            else:
                response_obj = response
                status_code = response.status_code if hasattr(response, 'status_code') else 200
            
            if isinstance(response_obj, str):
                response_obj = app.make_response(response_obj)
                status_code = response_obj.status_code
            
            logger.info(
                f"Completed request: {request.method} {request.path} "
                f"Status: {status_code} "
                f"Duration: {duration}ms"
            )
            
            return response
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
            
    return decorated_function

# Routes
@app.route('/')
@log_request
def index():
    return render_template('index.html')

@app.route('/prediction')
@log_request
def prediction():
    return render_template('prediction.html')

@app.route('/forecasting')
@log_request
def forecasting():
    return render_template('forecasting.html')

# Forecasting Functions
def get_historical_stream_data(content_id: str, days: int = 365) -> pd.DataFrame:
    """Retrieve historical stream data from DynamoDB with fallback."""
    if dynamodb is None:
        logger.error("DynamoDB not initialized")
        return None

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        response = table.query(
            KeyConditionExpression='content_id = :cid AND stream_date BETWEEN :start AND :end',
            ExpressionAttributeValues={
                ':cid': content_id,
                ':start': start_date.strftime('%Y-%m-%d'),
                ':end': end_date.strftime('%Y-%m-%d')
            }
        )

        if not response.get('Items'):
            logger.warning(f"No historical data found for content {content_id}")
            date_range = pd.date_range(start=start_date, end=end_date)
            return pd.DataFrame({
                'stream_date': date_range,
                'streams': [0] * len(date_range)
            })

        df = pd.DataFrame(response['Items'])
        df['stream_date'] = pd.to_datetime(df['stream_date'])
        df['streams'] = df['streams'].astype(int)
        df = df.sort_values('stream_date')

        date_range = pd.date_range(start=start_date, end=end_date)
        df = df.set_index('stream_date').reindex(date_range, fill_value=0).reset_index()
        df = df.rename(columns={'index': 'stream_date'})

        return df

    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        date_range = pd.date_range(end=datetime.now(), periods=days)
        return pd.DataFrame({
            'stream_date': date_range,
            'streams': np.random.randint(0, 50, size=days)
        })

def generate_forecast(historical_data: pd.DataFrame, periods: int = 30) -> dict:
    """Generate stream forecast using Holt-Winters method with fallback."""
    try:
        ts = historical_data.set_index('stream_date')['streams']
        
        model = ExponentialSmoothing(
            ts,
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            initialization_method='estimated'
        ).fit()
        
        forecast = model.forecast(periods)
        last_date = ts.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods
        )
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'values': forecast.round().astype(int).tolist(),
            'confidence_intervals': {
                'lower': (forecast * 0.8).round().astype(int).tolist(),
                'upper': (forecast * 1.2).round().astype(int).tolist()
            }
        }
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        # Generate dummy forecast
        forecast_dates = pd.date_range(
            start=datetime.now(),
            periods=periods
        )
        dummy_values = np.random.randint(5, 20, size=periods)
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'values': dummy_values.tolist(),
            'confidence_intervals': {
                'lower': (dummy_values * 0.8).astype(int).tolist(),
                'upper': (dummy_values * 1.2).astype(int).tolist()
            }
        }

@app.route('/api/predict', methods=['POST'])
@log_request
def predict():
    try:
        # Validate input
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        data = request.get_json()
        if not data:
            raise ValueError("Empty request body")
        
        required_fields = ['user_id', 'content_id']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        base_url = app.config['PREDICTION_API_URL'].rstrip('/')
        api_url = f"{base_url}/predict"

        headers = {
            'x-api-key': app.config['API_KEY'],
            'Content-Type': 'application/json'
        }

        logger.info(f"Calling API endpoint: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=app.config['REQUEST_TIMEOUT']
        )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            error_msg = f"API returned {response.status_code}"
            logger.error(f"{error_msg}: {response.text}")
            return jsonify({
                "error": error_msg,
                "api_response": response.text,
                "status": "error"
            }), 502

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error: {str(e)}")
        return jsonify({
            "error": "Prediction service unavailable",
            "details": str(e),
            "status": "error"
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "status": "error"
        }), 500

def validate_prediction_input(request):
    """Validate and extract prediction input from request."""
    if not request.is_json:
        raise ValueError("Request must be JSON")
    
    data = request.get_json()
    
    if not data:
        raise ValueError("Empty request body")
    
    # Validate required fields
    required_fields = ['user_id', 'content_id']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    return {
        "user_id": str(data['user_id']),
        "content_id": str(data['content_id']),
        "context": data.get('context', {})
    }

def make_api_request(url, payload, api_key, endpoint_name, max_retries=2):
    """Make an API request with exponential backoff retry."""
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json'
    }
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=app.config['REQUEST_TIMEOUT']
            )
            
            if response.status_code == 200:
                return response
                
            last_error = f"API returned {response.status_code}: {response.text}"
            
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            
        if attempt < max_retries:
            wait_time = (attempt + 1) * 0.5
            logger.warning(f"Retry {attempt + 1} for {endpoint_name} after {wait_time}s")
            time.sleep(wait_time)
    
    raise Exception(f"{endpoint_name} API failed after {max_retries} retries: {last_error}")


# API Endpoints
@app.route('/api/forecast', methods=['POST'])
@log_request
def forecast():
    try:
        data = validate_forecast_input(request)
        content_id = data['content_id']
        forecast_days = data['forecast_days']
        
        historical_data = get_historical_stream_data(content_id)
        forecast_result = generate_forecast(historical_data, forecast_days)

        # Calculate summary stats
        hist_summary = {
            'last_30_days': int(historical_data['streams'].tail(30).sum()),
            'last_90_days': int(historical_data['streams'].tail(90).sum()),
            'last_365_days': int(historical_data['streams'].sum())
        }
        
        return jsonify({
            'content_id': content_id,
            'forecast': forecast_result,
            'historical_summary': hist_summary,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'historical_summary': {
                'last_30_days': 0,
                'last_90_days': 0,
                'last_365_days': 0
            }
        }), 500

# Helper Functions
def validate_forecast_input(request):
    """Validate and extract forecast input from request."""
    if not request.is_json:
        raise ValueError("Request must be JSON")
    
    data = request.get_json()
    if not data:
        raise ValueError("Empty request body")
    
    if 'content_id' not in data:
        raise ValueError("Missing required field: content_id")
    
    forecast_days = int(data.get('forecast_days', 30))
    if forecast_days <= 0 or forecast_days > 365:
        raise ValueError("Forecast days must be between 1 and 365")
    
    return {
        "content_id": str(data['content_id']),
        "forecast_days": forecast_days
    }

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_HOST'), port=os.getenv('FLASK_PORT'), debug=os.getenv('FLASK_DEBUG'))
