# AWS Infrastructure Recommendation (CPU-Only Recommendation System)

## Scope
This system is a Python recommendation engine using:
- rule/heuristic filtering
- candidate generation
- scoring/ranking
- moderate pandas-style processing

It is CPU and memory bound. It does **not** need GPU today.

## Workload Split
1. Offline batch generation:
- thousands of users per run
- CPU-heavy per-user pipeline
- parallelizable across cores

2. Online serving:
- lightweight API/dashboard
- mostly read/format/return
- low compute, needs stable latency

## Recommended AWS Architecture
### Network + security baseline
- `VPC` with private subnets for app/data, public subnet only for ALB.
- `ALB` in front of API/dashboard instances.
- `Security Groups`:
  - ALB: 80/443 from internet
  - App: only from ALB on app port
  - DB/Redis: only from app SG
- `IAM Role` for instances/tasks (no static AWS keys in code).
- `Secrets Manager` or `SSM Parameter Store` for secrets/config.

### Online serving path
- Compute: `ECS Fargate` (or small EC2 ASG) for API/dashboard.
- Target size:
  - start with `2 vCPU / 4-8 GB RAM` per task/instance for API.
  - min 2 tasks across 2 AZs for HA.
- API container only reads generated bundle artifacts (S3/DB/cache), returns JSON.
- Put `CloudFront` in front only if global edge performance is needed.

### Offline batch path
- Preferred: `AWS Batch` on EC2 compute environment (best cost/perf for CPU-heavy jobs).
- Alternative: ECS scheduled tasks if jobs are simple and short.
- Instance families:
  - `c7i` / `c7a`: best for high single-thread and multi-core CPU tasks.
  - `m7i` / `m7a`: when memory headroom is more important.
  - `r7i` only if candidate pools become very memory-heavy.
- Initial sizing guidance:
  - small daily runs: `8 vCPU, 32 GB RAM`
  - medium: `16 vCPU, 64 GB RAM`
  - growth headroom: `32 vCPU, 128 GB RAM`
- Parallelization strategy:
  - shard users across multiple batch jobs by user-id hash/range.
  - keep each job deterministic with fixed seeds where needed.

### Storage
- Durable artifacts: `S3` (versioned bucket for generated bundles + run manifests).
- Runtime local disk:
  - EBS gp3 for EC2 root/data volumes (good default, predictable IOPS).
  - use instance NVMe only for temp scratch/intermediate files requiring very high local IO.
- If using ECS/Fargate only, keep artifacts in S3 (not ephemeral task storage).

### Data layer choice
- If serving can be file/object based:
  - store final generated outputs in S3 + optional Redis cache.
- If query flexibility/joins/filtering by user/time are needed:
  - use `RDS PostgreSQL` for final bundle records + metadata.
- Recommended practical setup:
  - S3 as source of truth artifacts.
  - RDS for indexed serving tables if API query patterns get complex.

### Caching
- `ElastiCache Redis` is useful for:
  - hot user bundle payloads
  - profile metadata lookups
  - reducing repeated DB/object reads
- Keep TTL-based cache invalidation per run_id/date partition.

## Service layout (reference)
- `ALB -> ECS service (API/dashboard) -> Redis (optional) -> RDS/S3`
- `EventBridge schedule -> AWS Batch job -> S3 (+ optional RDS upsert)`

## Cost-aware starting point
- Online:
  - ECS Fargate: 2 tasks x (2 vCPU, 4-8 GB)
- Batch:
  - AWS Batch on Spot-first `c7a/c7i` with On-Demand fallback
  - job sizes from 8-16 vCPU each, scale horizontally by shard count
- Storage:
  - S3 Standard for fresh outputs, lifecycle to IA/Glacier for older runs

## Observability
- CloudWatch logs/metrics:
  - per-stage timing (candidate gen, scoring, selection)
  - p50/p90/p95 latency
  - batch throughput (users/min)
  - error rates and retry counts
- Alarms:
  - API 5xx
  - batch failures
  - stale output age (no successful run in expected window)

## When GPU becomes useful later
GPU instances are justified only if you add workloads like:
- deep neural ranking models (PyTorch/TensorFlow inference/training)
- embedding generation at large scale with transformer models
- LLM-based reranking/content generation

Even then:
- serving might still remain CPU if model is small/quantized.
- use GPU first for training/offline embedding jobs, not automatically for API.

## Sensitive data policy (repo + AWS)
- Do not store AWS keys/tokens in repo.
- Use IAM roles for compute and managed secret stores for credentials.
- Keep raw user-identifiable exports out of git when not required for runtime.
- Store production data in S3/RDS with encryption at rest and least-privilege IAM.
