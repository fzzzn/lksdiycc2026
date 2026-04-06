# Machine Learning — Model Training

This directory contains the Jupyter notebook used to train the hybrid content recommendation model.

## Setup

Update the S3 bucket name in the notebook before running:

```python
# Replace with your actual S3 bucket
bucket_name = "your-bucket-name"
```

Update the `s3fs` library if needed:

```bash
pip install --upgrade s3fs
```

## Model Architecture

The training notebook implements three components:

| Component | Description |
|-----------|-------------|
| `ContentBasedRecommender` | TF-IDF similarity between content metadata (genre, type, studio) |
| `CollaborativeRecommender` | SVD matrix factorization on user-content interaction history |
| `HybridRecommender` | Weighted combination of content-based and collaborative scores |

## Training Pipeline

1. Load user profiles, content catalog, and interaction data from S3.
2. Train content-based and collaborative filtering models.
3. Combine into a hybrid recommender.
4. Save the trained model to S3 and update DynamoDB with content embeddings.

## Output

- Model artifact uploaded to S3 at the configured bucket and key path.
- Content embeddings stored in DynamoDB for use by the prediction Lambda.
