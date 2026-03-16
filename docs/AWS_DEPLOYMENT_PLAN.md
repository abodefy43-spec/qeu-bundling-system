# AWS Deployment Plan

## Scope

Deploy QEU as a CPU-first recommendation platform with two runtime roles:

- **Offline batch generation:** compute-heavy bundle generation
- **Online API serving:** lightweight read/format/return from precomputed artifacts

No GPU resources are required.

## Current Terraform Reality (`infra/terraform/main.tf`)

The current scaffold provisions foundational infrastructure:

- S3 artifact bucket (versioning + SSE + lifecycle)
- ECR repositories (optional)
- CloudWatch log groups
- IAM task roles and S3 access policy
- Security groups
- ECS cluster
- ALB, target group, and HTTP listener

This is a valid base layer. It does **not** yet define ECS task definitions, ECS service, or EventBridge schedule resources.

## Target Runtime Architecture

### API serving model

- API container runs on ECS/Fargate behind ALB
- API reads precomputed bundle artifacts
- API should not execute full bundle generation inline

### Batch model

- Batch container runs on ECS/Fargate schedule
- Batch writes run artifacts and metadata to S3
- New artifact version promoted only after validation

## Data and Artifact Flow

1. Scheduled batch run starts.
2. Bundles are generated and validated.
3. Artifacts are written to S3 under run-specific prefixes.
4. Latest pointer/manifest is updated.
5. API reads latest approved artifact set.

## Rollback Model

- S3 versioning is enabled.
- Rollback is done by reverting the latest pointer/manifest to a prior artifact version.
- Keep batch logs in CloudWatch to correlate rollbacks with failing runs.

## Networking and Security Baseline

- ALB in public subnets
- ECS workloads in private subnets (when task definitions are added)
- SG policy: ALB -> API port only
- IAM roles for runtime AWS access (no static keys)
- Secrets from Secrets Manager/SSM

## Observability

Minimum operational signals:

- API log group activity
- Batch log group activity
- ALB health check success
- Artifact freshness (time since last successful publish)
- Batch success/failure counts

## First Deployment Recommendation

### Phase 1 (now, with current `main.tf`)

Deploy the foundational layer in Terraform:

- S3
- ECR (optional)
- ECS cluster
- ALB stack
- IAM and log groups

### Phase 2 (next change set)

Add workload resources in Terraform:

- ECS task definition + ECS service for API image
- EventBridge schedule + ECS task target for batch image

### Phase 3 (only if needed)

Add optional services:

- Redis for hot-read caching
- PostgreSQL for relational query patterns

Do not add these before there is a measured need.
