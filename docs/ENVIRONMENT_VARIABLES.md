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
| `QEU_S3_FINAL_RECOMMENDATIONS_KEY` | S3 key for `final_recommendations_by_user.json` (batch upload + API bootstrap). | Optional | `output/final_recommendations_by_user.json` | No |
| `QEU_S3_FALLBACK_BUNDLE_BANK_KEY` | S3 key for `fallback_bundle_bank.json` generated in batch materialization. | Optional | `output/fallback_bundle_bank.json` | No |
| `QEU_API_LOG_LEVEL` | API server log level. | Optional | `INFO` | No |

## Gunicorn Runtime (API Container)

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `GUNICORN_WORKER_CLASS` | Gunicorn worker class for API serving. | Optional | `gthread` | No |
| `GUNICORN_WORKERS` | Gunicorn worker count (memory-safe default is `1`). | Optional | `1` | No |
| `GUNICORN_THREADS` | Threads per worker. | Optional | `4` | No |
| `GUNICORN_TIMEOUT` | Worker timeout in seconds. | Optional | `120` | No |

## Batch-Only

| Name | Purpose | Required | Example | Secret |
|---|---|---|---|---|
| `QEU_SERVING_PROFILE` | Enables serving profiling output. | Optional | `0` | No |
| `QEU_FINAL_RECOMMENDATIONS_MAX_USERS` | Cap final recommendation materialization to N users (test mode). Unset/empty/`0`/`off`/`unlimited` = full uncapped user set. | Optional | _(unset)_ | No |
| `QEU_FINAL_RECOMMENDATIONS_USER_SELECTION` | User selection mode when `QEU_FINAL_RECOMMENDATIONS_MAX_USERS` is set. `sorted` or `random`. | Optional | `sorted` | No |
| `QEU_FINAL_RECOMMENDATIONS_RANDOM_SEED` | Optional random seed for reproducible `random` selection mode. | Optional | `42` | No |
| `QEU_FALLBACK_BUNDLE_BANK_ENABLED` | Enable offline fallback bank generation/backfill during final materialization. | Optional | `1` | No |
| `QEU_FALLBACK_BUNDLE_BANK_TARGET_SIZE` | Target number of globally ranked fallback bundles to keep. | Optional | `1000` | No |
| `QEU_FALLBACK_BUNDLE_BANK_MAX_SIZE` | Hard upper cap for fallback bank size. | Optional | `2000` | No |
| `QEU_FALLBACK_BUNDLE_MIN_SCORE` | Optional minimum fallback quality score threshold. Unset uses rule-based filtering only. | Optional | `50` | No |
| `QEU_S3_FILTERED_ORDERS_KEY` | S3 key for `filtered_orders.pkl` used by `run materialize-final`. | Optional | `processed/filtered_orders.pkl` | No |
| `QEU_S3_SCORED_CANDIDATES_KEY` | S3 key for `person_candidates_scored.csv` used by `run materialize-final`. | Optional | `output/person_candidates_scored.csv` | No |
| `QEU_S3_CANDIDATE_PAIRS_KEY` | S3 key for `person_candidate_pairs.csv` fallback used by `run materialize-final`. | Optional | `processed/candidates/person_candidate_pairs.csv` | No |

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
