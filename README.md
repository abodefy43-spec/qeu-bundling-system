# QEU Product Bundling System

QEU is a Python recommendation system for retail product bundles. The workload is CPU-bound and rule/heuristic driven.

## Runtime Model

- **Batch generation (heavy):** builds candidates, applies rules and scoring, and writes recommendation artifacts.
- **API/dashboard serving (light):** reads precomputed artifacts and returns responses.
- **Dependency split:** `requirements.api.txt` for serving image, `requirements.batch.txt` for batch image.

The API path is intended for serving precomputed results, not full inline recommendation generation.

## Local Development

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run quick batch pipeline

```bash
python -m qeu_bundling.cli run quick
```

### Run API/dashboard

```bash
python -m qeu_bundling.cli serve --host 127.0.0.1 --port 5000
```

## AWS Terraform Deployment (Current State)

Current `infra/terraform/main.tf` provisions:

- S3 artifact bucket (versioning, encryption, lifecycle)
- Optional ECR repositories for API and batch images
- CloudWatch log groups
- IAM roles and S3 access policy for ECS tasks
- Security groups
- ECS cluster
- ALB, target group, and HTTP listener
- ECS API task definition + ECS service (Fargate)
- ECS batch task definition + EventBridge schedule target

## Why CPU-Only

QEU’s core workload is Python control flow, filtering, ranking, and rules. This scaffold does not require GPU infrastructure.

## Deployment References

- Architecture rationale: `docs/AWS_INFRA_V1.md`
- Deployment plan: `docs/AWS_DEPLOYMENT_PLAN.md`
- Environment variables: `docs/ENVIRONMENT_VARIABLES.md`
- Terraform runbook: `infra/terraform/README.md`
- API Dockerfile: `Dockerfile.api`
- Batch Dockerfile: `Dockerfile.batch`
