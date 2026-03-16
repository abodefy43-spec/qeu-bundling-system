# Terraform Runbook

This directory contains the AWS foundation stack for QEU.

## What current `main.tf` provisions

- S3 artifacts bucket:
  - versioning enabled
  - server-side encryption enabled
  - lifecycle transition to Standard-IA
  - public access blocked
- Optional ECR repositories for API and batch images
- CloudWatch log groups for API and batch workloads
- IAM roles/policies for ECS task execution and S3 access
- Security groups for ALB and API ingress policy
- ECS cluster
- ALB, target group, and HTTP listener

## What current `main.tf` does **not** provision

- ECS task definitions
- ECS service
- EventBridge schedule and ECS run target
- RDS/Redis resources

Treat those as future changes, not part of this stack today.

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
terraform plan
terraform apply
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

## Security notes

- Keep secrets out of git and out of committed `terraform.tfvars`.
- Use IAM roles and AWS Secrets Manager/SSM for sensitive values.
