# Lambda — Stream Forecasting

This Lambda function forecasts the number of streams for a given content item or content type using time-series methods.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECASTING_MODEL_BUCKET` | — | S3 bucket containing the model artifact |
| `FORECASTING_MODEL_KEY` | `models/forecasting_model.pkl` | S3 key for the model file |
| `CONTENT_EMBEDDINGS_TABLE` | `ContentEmbeddings` | DynamoDB table for content data |
| `STREAM_HISTORY_TABLE` | `StreamHistory` | DynamoDB table for stream history |
| `USER_INTERACTIONS_TABLE` | `UserInteractions` | DynamoDB table for user interactions |

## Request Format

**Method:** POST

```json
{
  "content_id": "content_00001",
  "content_type": "Movie",
  "method": "moving_average",
  "periods": 30,
  "metric": "streams"
}
```

### Parameters

| Field | Required | Default | Options |
|-------|----------|---------|---------|
| `content_id` | No | — | Any valid content ID |
| `content_type` | No | — | `Movie`, `Series`, `Music`, `Podcast` |
| `method` | No | `moving_average` | `moving_average`, `exponential_smoothing`, `linear_trend`, `seasonal` |
| `periods` | No | `30` | 1–365 |
| `metric` | No | `streams` | `streams`, `watch_duration` |

## Response Format

```json
{
  "forecast": [320, 415, 388],
  "forecast_dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
  "method": "moving_average",
  "confidence": "high",
  "metric": "streams",
  "periods": 30,
  "summary": {
    "total_forecast": 10500,
    "average_daily": 350.0,
    "min_daily": 210,
    "max_daily": 520,
    "std_deviation": 65.3
  },
  "historical_data_points": 72,
  "historical_average": 330.0,
  "timestamp": "2024-01-01T00:00:00",
  "status": "success"
}
```
