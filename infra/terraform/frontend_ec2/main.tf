provider "aws" {
  region = var.region
}

resource "aws_instance" "frontend" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = var.public_subnet_id
  vpc_security_group_ids = [var.frontend_sg_id]

  user_data = file("${path.module}/user_data/setup_swap.sh")

  tags = {
    Name    = "frontend-ec2"
    Role    = "frontend"
  }
}

######################

resource "aws_iam_role" "frontend_ec2" {
  name = "frontend-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_instance_profile" "frontend_ec2" {
  name = "frontend-ec2-profile"
  role = aws_iam_role.frontend_ec2.name
}

resource "aws_eip" "frontend" {
  instance = aws_instance.frontend.id
  domain   = "vpc"
}

in EC2:
iam_instance_profile = aws_iam_instance_profile.frontend_ec2.name


############ 

gehören in VPC-Stack
- Routing_Table
- IGW