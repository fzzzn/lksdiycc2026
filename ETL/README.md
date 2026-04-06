# ETL — AWS Glue Spark Job

This script runs as an AWS Glue job. It reads raw data from the Glue Data Catalog and writes processed feature datasets to S3.

## Configuration

Before deploying, update the S3 output path in the script:

```python
# Replace with your actual S3 bucket
"s3://yourbucket/processed-data/..."
```

## Output Datasets

| Dataset | S3 Path | Description |
|---------|---------|-------------|
| User-content matrix | `processed-data/user_content_matrix/` | Stream counts and ratings per user-content pair |
| Content statistics | `processed-data/content_stats/` | Viewer counts and average rating per content item |
| User features | `processed-data/user_features/` | Behavioral features joined with user profiles |

## Source Tables (Glue Data Catalog)

- `streamify_database.user_profiles`
- `streamify_database.user_interactions`
- `streamify_database.content_catalog`
