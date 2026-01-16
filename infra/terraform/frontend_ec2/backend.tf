terraform {
  backend "s3" {
    bucket = "backend-terraform-state-rf"
    key    = "frontend-ec2/terraform.tfstate"
    region = "eu-north-1"
  }
}
