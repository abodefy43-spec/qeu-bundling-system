variable "project_name" {
  description = "Project slug used in resource names."
  type        = string
  default     = "qeu-bundling"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "project_name must use lowercase letters, digits, and hyphens only."
  }
}

variable "environment" {
  description = "Deployment environment name (for example: staging, prod)."
  type        = string
  default     = "staging"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.environment))
    error_message = "environment must use lowercase letters, digits, and hyphens only."
  }
}

variable "aws_region" {
  description = "AWS region for provider operations."
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC ID where resources will be created."
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for ALB placement."
  type        = list(string)

  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "Provide at least two public subnets for multi-AZ ALB placement."
  }
}

variable "allowed_ingress_cidrs" {
  description = "CIDR blocks allowed to reach the ALB listener on port 80."
  type        = list(string)
  default     = ["0.0.0.0/0"]

  validation {
    condition     = alltrue([for c in var.allowed_ingress_cidrs : can(cidrhost(c, 0))])
    error_message = "Each allowed_ingress_cidrs entry must be a valid CIDR block."
  }
}

variable "artifacts_bucket_name" {
  description = "Optional explicit S3 artifacts bucket name. Leave empty for auto-generated name."
  type        = string
  default     = ""
}

variable "s3_force_destroy" {
  description = "Allow destroying a non-empty artifacts bucket. Use false in production."
  type        = bool
  default     = false
}

variable "s3_transition_to_ia_days" {
  description = "Days before objects transition to S3 Standard-IA."
  type        = number
  default     = 30

  validation {
    condition     = var.s3_transition_to_ia_days >= 1
    error_message = "s3_transition_to_ia_days must be at least 1."
  }
}

variable "create_ecr_repositories" {
  description = "Create ECR repositories for API and batch images."
  type        = bool
  default     = true
}

variable "api_image" {
  description = "Explicit API image URI. If empty, use managed ECR repository URL."
  type        = string
  default     = ""
}

variable "batch_image" {
  description = "Explicit batch image URI. If empty, use managed ECR repository URL."
  type        = string
  default     = ""
}

variable "api_container_port" {
  description = "Port exposed by the API container/target group."
  type        = number
  default     = 5000

  validation {
    condition     = var.api_container_port >= 1 && var.api_container_port <= 65535
    error_message = "api_container_port must be between 1 and 65535."
  }
}

variable "api_health_check_path" {
  description = "HTTP path used by ALB health checks."
  type        = string
  default     = "/readyz"

  validation {
    condition     = can(regex("^/", var.api_health_check_path))
    error_message = "api_health_check_path must start with '/'."
  }
}

variable "api_health_check_timeout_seconds" {
  description = "ALB target-group health check timeout in seconds for API tasks."
  type        = number
  default     = 10

  validation {
    condition     = var.api_health_check_timeout_seconds >= 2 && var.api_health_check_timeout_seconds <= 120
    error_message = "api_health_check_timeout_seconds must be between 2 and 120."
  }
}

variable "common_env" {
  description = "Shared environment variables map (currently used by locals in main.tf)."
  type        = map(string)
  default     = {}
}

variable "api_env" {
  description = "API-specific environment variables map (currently used by locals in main.tf)."
  type        = map(string)
  default     = {}
}

variable "batch_env" {
  description = "Batch-specific environment variables map (currently used by locals in main.tf)."
  type        = map(string)
  default     = {}
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for ECS tasks (API service and scheduled batch)."
  type        = list(string)

  validation {
    condition     = length(var.private_subnet_ids) >= 1
    error_message = "Provide at least one private subnet ID."
  }
}

variable "api_task_cpu" {
  description = "Fargate CPU units for the API task definition."
  type        = number
  default     = 1024
}

variable "api_task_memory" {
  description = "Fargate memory (MiB) for the API task definition."
  type        = number
  default     = 4096
}

variable "api_desired_count" {
  description = "Desired number of API tasks in the ECS service."
  type        = number
  default     = 2
}

variable "api_health_check_grace_period_seconds" {
  description = "Grace period for ECS service target health checks during startup."
  type        = number
  default     = 180

  validation {
    condition     = var.api_health_check_grace_period_seconds >= 0 && var.api_health_check_grace_period_seconds <= 7200
    error_message = "api_health_check_grace_period_seconds must be between 0 and 7200."
  }
}

variable "api_assign_public_ip" {
  description = "Assign public IP to API tasks. Keep false when private subnets have NAT."
  type        = bool
  default     = false
}

variable "api_command" {
  description = "Deprecated. Kept only for backwards compatibility and intentionally ignored."
  type        = list(string)
  default     = []
}

variable "api_gunicorn_worker_class" {
  description = "Gunicorn worker class used by the API ECS task."
  type        = string
  default     = "gthread"
}

variable "api_gunicorn_workers" {
  description = "Gunicorn worker count for API serving."
  type        = number
  default     = 1

  validation {
    condition     = var.api_gunicorn_workers >= 1 && var.api_gunicorn_workers <= 8
    error_message = "api_gunicorn_workers must be between 1 and 8."
  }
}

variable "api_gunicorn_threads" {
  description = "Gunicorn thread count per worker for API serving."
  type        = number
  default     = 4

  validation {
    condition     = var.api_gunicorn_threads >= 1 && var.api_gunicorn_threads <= 32
    error_message = "api_gunicorn_threads must be between 1 and 32."
  }
}

variable "api_gunicorn_timeout_seconds" {
  description = "Gunicorn worker timeout for API serving."
  type        = number
  default     = 120

  validation {
    condition     = var.api_gunicorn_timeout_seconds >= 30 && var.api_gunicorn_timeout_seconds <= 600
    error_message = "api_gunicorn_timeout_seconds must be between 30 and 600."
  }
}

variable "batch_task_cpu" {
  description = "Fargate CPU units for the batch task definition."
  type        = number
  default     = 2048
}

variable "batch_task_memory" {
  description = "Fargate memory (MiB) for the batch task definition."
  type        = number
  default     = 4096
}

variable "batch_command" {
  description = "Command used by the batch container in ECS scheduled runs."
  type        = list(string)
  default     = ["python", "-m", "qeu_bundling.cli", "run", "full"]
}

variable "batch_task_count" {
  description = "Number of batch tasks to launch per EventBridge schedule trigger."
  type        = number
  default     = 1
}

variable "batch_schedule_expression" {
  description = "EventBridge cron/rate expression for scheduled batch runs."
  type        = string
  default     = "cron(0 2 * * ? *)"
}

variable "batch_schedule_enabled" {
  description = "Enable or disable the EventBridge batch schedule."
  type        = bool
  default     = true
}

variable "batch_assign_public_ip" {
  description = "Assign public IP to scheduled batch tasks."
  type        = bool
  default     = false
}
