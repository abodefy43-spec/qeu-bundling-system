output "alb_dns_name" {
  description = "DNS name of the application load balancer."
  value       = aws_lb.api.dns_name
}

output "alb_arn" {
  description = "ARN of the application load balancer."
  value       = aws_lb.api.arn
}

output "alb_listener_http_arn" {
  description = "ARN of the ALB HTTP listener."
  value       = aws_lb_listener.api_http.arn
}

output "api_target_group_arn" {
  description = "ARN of the ALB target group for the API."
  value       = aws_lb_target_group.api.arn
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster."
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster."
  value       = aws_ecs_cluster.main.arn
}

output "artifacts_bucket_name" {
  description = "Name of the S3 artifacts bucket."
  value       = aws_s3_bucket.artifacts.bucket
}

output "artifacts_bucket_arn" {
  description = "ARN of the S3 artifacts bucket."
  value       = aws_s3_bucket.artifacts.arn
}

output "api_log_group_name" {
  description = "CloudWatch log group name for API workload logs."
  value       = aws_cloudwatch_log_group.api.name
}

output "batch_log_group_name" {
  description = "CloudWatch log group name for batch workload logs."
  value       = aws_cloudwatch_log_group.batch.name
}

output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role."
  value       = aws_iam_role.ecs_task_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role with S3 artifact access policy."
  value       = aws_iam_role.ecs_task_role.arn
}

output "alb_security_group_id" {
  description = "Security group ID for the ALB."
  value       = aws_security_group.alb.id
}

output "ecs_api_security_group_id" {
  description = "Security group ID intended for API ECS tasks."
  value       = aws_security_group.ecs_api.id
}

output "api_ecr_repository_url" {
  description = "API ECR repository URL when create_ecr_repositories=true."
  value       = var.create_ecr_repositories ? aws_ecr_repository.api[0].repository_url : ""
}

output "batch_ecr_repository_url" {
  description = "Batch ECR repository URL when create_ecr_repositories=true."
  value       = var.create_ecr_repositories ? aws_ecr_repository.batch[0].repository_url : ""
}

output "api_task_definition_arn" {
  description = "ARN of the API ECS task definition."
  value       = aws_ecs_task_definition.api.arn
}

output "batch_task_definition_arn" {
  description = "ARN of the batch ECS task definition."
  value       = aws_ecs_task_definition.batch.arn
}

output "api_service_name" {
  description = "Name of the ECS API service."
  value       = aws_ecs_service.api.name
}

output "api_service_id" {
  description = "ID of the ECS API service."
  value       = aws_ecs_service.api.id
}

output "eventbridge_batch_rule_name" {
  description = "Name of the EventBridge rule that triggers the batch task."
  value       = aws_cloudwatch_event_rule.batch_schedule.name
}

output "eventbridge_batch_rule_arn" {
  description = "ARN of the EventBridge rule that triggers the batch task."
  value       = aws_cloudwatch_event_rule.batch_schedule.arn
}

output "eventbridge_invoke_role_arn" {
  description = "ARN of the IAM role EventBridge uses to run ECS batch tasks."
  value       = aws_iam_role.eventbridge_invoke_ecs.arn
}
