variable "region" {
  type    = string
  default = "eu-north-1"
}

variable "vpc_id" {
    type = string
}

variable "public_subnet_id" {
    type = string
}

variable "frontend_sg_id" {
    type = string
}

variable "instance_type" {
  type    = string
  default = "t3.small"
}

variable "ami_id" {
  type        = string
  description = "Custom frontend AMI"
}

variable "key_name" {
  type        = string
  description = "Existing EC2 key pair"
}

variable "ssh_cidr" {
  type    = string
  default = "0.0.0.0/0"
}
