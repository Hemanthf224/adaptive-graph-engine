# ---------------------------------------------------------
# Phase 42: Cloud-Native Global Deployment
# Terraform configuration for AWS Elastic Kubernetes Service (EKS)
# ---------------------------------------------------------

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# 1. VPC Configuration for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "adaptive-graph-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

# 2. AWS EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.15.3"

  cluster_name    = "adaptive-graph-cluster"
  cluster_version = "1.27"

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.public_subnets

  # 3. Managed Node Groups (Auto-Scaling HPC Nodes)
  eks_managed_node_groups = {
    # Standard compute nodes for Web/Frontend
    frontend_nodes = {
      min_size     = 2
      max_size     = 10
      desired_size = 2
      instance_types = ["t3.large"]
    }
    
    # Massive GPU/Compute nodes for the C++ Graph AI Engine
    hpc_backend_nodes = {
      min_size     = 1
      max_size     = 50
      desired_size = 3
      instance_types = ["c6i.24xlarge"] # 96 vCPUs for OpenMP parallelization
      
      # Taint the nodes so only the backend engine pods can run here
      taints = {
        dedicated = {
          key    = "workload"
          value  = "hpc"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}
