# Lambda — Stream Prediction

This Lambda function predicts the probability that a user will stream a given content, using a hybrid recommendation model.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_BUCKET` | — | S3 bucket containing the model artifact |
| `MODEL_KEY` | `models/hybrid_model.pkl` | S3 key for the model file |
| `USERS_TABLE` | `streamify-users` | DynamoDB table for user profiles |
| `CONTENT_TABLE` | `ContentEmbeddings` | DynamoDB table for content features |

## Request Format

**Method:** POST

```json
{
  "user_id": "user_00001",
  "content_id": "content_00001"
}
```

## Response Format

```json
{
  "prediction": 0.74,
  "prediction_percentage": 74.0,
  "model_source": "ml_model",
  "timestamp": "2024-01-01T00:00:00",
  "status": "success",
  "user_features": {
    "stream_count": 85,
    "subscription_plan": "Premium",
    "total_watch_hours": 120.5
  },
  "content_features": {
    "avg_rating": 4.6,
    "content_type": "Series",
    "is_exclusive": true,
    "popularity_score": 0.88
  }
}
```

The `model_source` field indicates whether the prediction came from the loaded ML model (`ml_model`) or a fallback heuristic (`fallback`).
