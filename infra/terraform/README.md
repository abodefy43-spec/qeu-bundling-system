# Terraform Runbook

This directory contains the AWS infrastructure for QEU API + scheduled batch on ECS/Fargate.

## What `main.tf` provisions

- S3 artifacts bucket:
  - versioning enabled
  - server-side encryption enabled
  - lifecycle transition to Standard-IA
  - public access blocked
- Optional ECR repositories for API and batch images
- CloudWatch log groups for API and batch workloads
- IAM roles/policies for ECS task execution, task runtime, and EventBridge invoke
- Security groups for ALB and API ingress policy
- ECS cluster
- ALB, target group, and HTTP listener
- ECS task definitions for API and batch
- ECS service for API on Fargate
- EventBridge scheduled rule + ECS batch target

## API Readiness + Memory-Safe Defaults

- ALB target health check path defaults to `/readyz` (not `/healthz`).
- API task defaults:
  - `api_task_cpu = 4096`
  - `api_task_memory = 16384`
  - `api_desired_count = 1` (set higher only when you need extra replica capacity)
- Health check controls:
  - `api_health_check_timeout_seconds = 10`
  - `api_health_check_grace_period_seconds = 180`
- Gunicorn runtime should use one worker by default (`gthread`, `workers=1`, `threads=4`, `--preload=true`).

## Runtime Separation

- API is an always-on ECS **service** (`aws_ecs_service.api`) and should run 24/7.
- Batch is an ECS **task definition** invoked by EventBridge (`aws_cloudwatch_event_target.batch_task`), not a service.
- Each scheduled batch run is one-off: it runs, writes artifacts, uploads to S3, then exits.

## Batch Modes

- Production daily mode (full compute): set `batch_command = ["python", "-m", "qeu_bundling.cli", "run", "full"]`.
- Fast test mode (10 users, same materialization flow): run one-off task override with:

```bash
python -m qeu_bundling.cli run materialize-final --max-users 10
```

## Prerequisites

- Terraform `>= 1.6`
- AWS credentials with permissions for the resources above
- Existing VPC and at least two public subnets

## Configure variables

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit required values in `terraform.tfvars`:

- `vpc_id`
- `public_subnet_ids`
- `allowed_ingress_cidrs`
- image settings (`api_image`, `batch_image`) or `create_ecr_repositories = true`

## Initialize and deploy

```bash
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## Build and push container images

```bash
docker build -f Dockerfile.api -t <registry>/qeu-api:<tag> .
docker build -f Dockerfile.batch -t <registry>/qeu-batch:<tag> .
docker push <registry>/qeu-api:<tag>
docker push <registry>/qeu-batch:<tag>
```

Image dependency profiles:

- `Dockerfile.api` installs `requirements.api.txt` (lean serving stack).
- `Dockerfile.batch` installs `requirements.batch.txt` (full generation/training stack).

If ECR repositories were created by Terraform, tag and push to the output repository URLs.

## Verify deployed foundation resources

```bash
terraform output
```

Check:

- ALB DNS is present
- target group and listener are present
- S3 artifacts bucket exists with versioning
- CloudWatch log groups exist
- IAM roles/policies are attached
- API service/task definition outputs are present
- EventBridge batch rule outputs are present

## Security notes

- Keep secrets out of git and out of committed `terraform.tfvars`.
- Use IAM roles and AWS Secrets Manager/SSM for sensitive values.
