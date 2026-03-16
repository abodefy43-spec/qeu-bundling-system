# Environment Variables

Reference for runtime configuration in local, staging, and production.

## Handling Rules

- Local: `.env` or shell export is acceptable.
- Staging/production: inject via ECS task definition and pull sensitive values from Secrets Manager or SSM.
- Never commit secrets.

## Shared / Common

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `QEU_PROJECT_ROOT` | Base path for project data/output resolution. | Optional | `/app` | No |
| `QEU_DATA_RAW_DIR` | Raw data directory override. | Optional | `/app/data/raw` | No |
| `QEU_DATA_PROCESSED_DIR` | Processed data directory override. | Optional | `/app/data/processed` | No |
| `QEU_DATA_REFERENCE_DIR` | Reference data directory override. | Optional | `/app/data/reference` | No |
| `QEU_OUTPUT_DIR` | Output artifacts directory override. | Optional | `/app/output` | No |
| `AWS_REGION` | AWS SDK region selection. | Recommended on AWS | `us-east-1` | No |
| `QEU_ARTIFACTS_S3_BUCKET` | S3 bucket for bundle artifacts. | Recommended on AWS | `qeu-bundling-staging-artifacts` | No |

## API-Only

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `FLASK_SECRET_KEY` | Flask session signing key. | Required in staging/prod | `replace-with-random-secret` | Yes |
| `QEU_FLASK_HOST` | Host bind when running Flask directly. | Optional | `0.0.0.0` | No |
| `QEU_FLASK_PORT` | Port bind when running Flask directly. | Optional | `5000` | No |
| `QEU_LOCAL_FAST_MODE` | Local dashboard shortcut mode. Keep disabled in production. | Optional | `0` | No |
| `QEU_DASHBOARD_DEFAULT_PERSON_COUNT` | Default profile count shown in dashboard. | Optional | `10` | No |

## Batch-Only

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `QEU_SERVING_PROFILE` | Enables serving profiling output. | Optional | `0` | No |

## Shared Recommendation Flags

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `QEU_USE_NEW_BUNDLE_SEMANTICS` | Toggle semantic recommendation path. | Optional | `1` | No |
| `QEU_STRICT_SEMANTIC_FILTERING` | Toggle strict semantic filtering. | Optional | `1` | No |
| `QEU_ENABLE_INTERNAL_STAPLES` | Toggle internal staples behavior. | Optional | `1` | No |
| `QEU_ENABLE_STAPLES_LANE` | Toggle optional staples lane behavior. | Optional | `0` | No |

## Future Optional Data Store Variables

Use these only if future infrastructure adds the corresponding resources:

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `DATABASE_URL` | PostgreSQL connection URL. | Optional | `postgresql://user:pass@host:5432/db` | Yes |
| `REDIS_URL` | Redis connection URL. | Optional | `redis://host:6379/0` | Usually |
