########################################
# AWS ACCOUNT + REGION
########################################

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

########################################
# LOCALS
########################################

locals {

  name_prefix = "${var.project_name}-${var.environment}"

  api_container_command = concat(
    [
      "gunicorn",
      "--bind", "0.0.0.0:${var.api_container_port}",
      "--worker-class", var.api_gunicorn_worker_class,
      "--workers", tostring(var.api_gunicorn_workers),
      "--threads", tostring(var.api_gunicorn_threads),
      "--timeout", tostring(var.api_gunicorn_timeout_seconds),
    ],
    var.api_gunicorn_preload ? ["--preload"] : [],
    ["qeu_bundling.api.server:app"],
  )

  artifacts_bucket_name = var.artifacts_bucket_name != "" ? var.artifacts_bucket_name : lower(
    replace(
      "${var.project_name}-${var.environment}-${data.aws_caller_identity.current.account_id}-${data.aws_region.current.name}-artifacts",
      "_",
      "-"
    )
  )

  api_container_env = [
    for k, v in merge(
      var.common_env,
      var.api_env,
      {
        QEU_PROJECT_ROOT        = "/app"
        QEU_ARTIFACTS_S3_BUCKET = aws_s3_bucket.artifacts.bucket
      }
    ) :
    {
      name  = k
      value = v
    }
  ]

  batch_container_env = [
    for k, v in merge(
      var.common_env,
      var.batch_env,
      {
        QEU_PROJECT_ROOT        = "/app"
        QEU_ARTIFACTS_S3_BUCKET = aws_s3_bucket.artifacts.bucket
      }
    ) :
    {
      name  = k
      value = v
    }
  ]

}

########################################
# S3 ARTIFACT BUCKET
########################################

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.artifacts_bucket_name
  force_destroy = var.s3_force_destroy
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "transition-to-standard-ia"
    status = "Enabled"

    filter {}

    transition {
      days          = var.s3_transition_to_ia_days
      storage_class = "STANDARD_IA"
    }
  }
}

########################################
# ECR REPOSITORIES
########################################

resource "aws_ecr_repository" "api" {
  count = var.create_ecr_repositories ? 1 : 0

  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "batch" {
  count = var.create_ecr_repositories ? 1 : 0

  name                 = "${local.name_prefix}-batch"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

########################################
# CLOUDWATCH LOG GROUPS
########################################

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name_prefix}/api"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/ecs/${local.name_prefix}/batch"
  retention_in_days = 30
}

########################################
# IAM ROLES
########################################

data "aws_iam_policy_document" "ecs_task_assume_role" {

  statement {

    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }

  }
}

resource "aws_iam_role" "ecs_task_execution" {

  name               = "${local.name_prefix}-ecs-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json

}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {

  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"

}

resource "aws_iam_role" "ecs_task_role" {

  name               = "${local.name_prefix}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume_role.json

}

########################################
# S3 ACCESS POLICY FOR TASKS
########################################

data "aws_iam_policy_document" "ecs_task_policy" {

  statement {

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ]

    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*"
    ]

  }

}

resource "aws_iam_role_policy" "ecs_task_inline" {

  name   = "${local.name_prefix}-ecs-task-policy"
  role   = aws_iam_role.ecs_task_role.id
  policy = data.aws_iam_policy_document.ecs_task_policy.json

}

########################################
# SECURITY GROUPS
########################################

resource "aws_security_group" "alb" {

  name        = "${local.name_prefix}-alb-sg"
  description = "ALB security group"
  vpc_id      = var.vpc_id

  ingress {

    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs

  }

  egress {

    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]

  }

}

resource "aws_security_group" "ecs_api" {

  name        = "${local.name_prefix}-ecs-api-sg"
  description = "API security group"
  vpc_id      = var.vpc_id

  ingress {

    from_port       = var.api_container_port
    to_port         = var.api_container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]

  }

  egress {

    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]

  }

}

########################################
# ECS CLUSTER
########################################

resource "aws_ecs_cluster" "main" {

  name = "${local.name_prefix}-cluster"

}

########################################
# APPLICATION LOAD BALANCER
########################################

resource "aws_lb" "api" {

  name               = "${local.name_prefix}-alb"
  load_balancer_type = "application"
  internal           = false

  security_groups = [aws_security_group.alb.id]
  subnets         = var.public_subnet_ids

}

resource "aws_lb_target_group" "api" {

  name        = "${local.name_prefix}-tg"
  port        = var.api_container_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {

    path                = var.api_health_check_path
    interval            = 30
    timeout             = var.api_health_check_timeout_seconds
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200-399"

  }

}

resource "aws_lb_listener" "api_http" {

  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {

    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn

  }

}

########################################
# ECS TASK DEFINITIONS
########################################

resource "aws_ecs_task_definition" "api" {

  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.api_task_cpu)
  memory                   = tostring(var.api_task_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.api_image
      essential = true
      command   = local.api_container_command

      portMappings = [
        {
          containerPort = var.api_container_port
          hostPort      = var.api_container_port
          protocol      = "tcp"
        }
      ]

      environment = local.api_container_env

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = data.aws_region.current.name
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  lifecycle {
    precondition {
      condition     = trimspace(var.api_image) != ""
      error_message = "api_image is empty. Set var.api_image to a full URI including tag (for example :staging)."
    }
  }

}

resource "aws_ecs_task_definition" "batch" {

  family                   = "${local.name_prefix}-batch"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.batch_task_cpu)
  memory                   = tostring(var.batch_task_memory)
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "batch"
      image     = var.batch_image
      essential = true
      command   = var.batch_command

      environment = local.batch_container_env

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.batch.name
          awslogs-region        = data.aws_region.current.name
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  lifecycle {
    precondition {
      condition     = trimspace(var.batch_image) != ""
      error_message = "batch_image is empty. Set var.batch_image to a full URI including tag (for example :staging)."
    }
  }

}

########################################
# ECS API SERVICE
########################################

resource "aws_ecs_service" "api" {

  name            = "${local.name_prefix}-api-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = var.api_health_check_grace_period_seconds

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_api.id]
    assign_public_ip = var.api_assign_public_ip
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = var.api_container_port
  }

  depends_on = [aws_lb_listener.api_http]

}

########################################
# EVENTBRIDGE SCHEDULE FOR BATCH TASK
########################################

data "aws_iam_policy_document" "eventbridge_assume_role" {

  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }

}

resource "aws_iam_role" "eventbridge_invoke_ecs" {

  name               = "${local.name_prefix}-events-invoke-ecs"
  assume_role_policy = data.aws_iam_policy_document.eventbridge_assume_role.json

}

data "aws_iam_policy_document" "eventbridge_invoke_ecs_policy" {

  statement {
    actions = ["ecs:RunTask"]
    resources = [
      aws_ecs_task_definition.batch.arn,
      "${aws_ecs_task_definition.batch.arn_without_revision}:*",
    ]
  }

  statement {
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.ecs_task_execution.arn,
      aws_iam_role.ecs_task_role.arn,
    ]
  }

}

resource "aws_iam_role_policy" "eventbridge_invoke_ecs" {

  name   = "${local.name_prefix}-events-invoke-ecs"
  role   = aws_iam_role.eventbridge_invoke_ecs.id
  policy = data.aws_iam_policy_document.eventbridge_invoke_ecs_policy.json

}

resource "aws_cloudwatch_event_rule" "batch_schedule" {

  name                = "${local.name_prefix}-batch-schedule"
  description         = "Scheduled batch recommendation generation"
  schedule_expression = var.batch_schedule_expression
  state               = var.batch_schedule_enabled ? "ENABLED" : "DISABLED"

}

resource "aws_cloudwatch_event_target" "batch_task" {

  rule      = aws_cloudwatch_event_rule.batch_schedule.name
  target_id = "batch-task"
  arn       = aws_ecs_cluster.main.arn
  role_arn  = aws_iam_role.eventbridge_invoke_ecs.arn

  ecs_target {
    launch_type         = "FARGATE"
    platform_version    = "LATEST"
    task_count          = var.batch_task_count
    task_definition_arn = aws_ecs_task_definition.batch.arn

    network_configuration {
      subnets          = var.private_subnet_ids
      security_groups  = [aws_security_group.ecs_api.id]
      assign_public_ip = var.batch_assign_public_ip
    }
  }

  depends_on = [aws_iam_role_policy.eventbridge_invoke_ecs]

}
