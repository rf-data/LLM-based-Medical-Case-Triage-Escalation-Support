```
cd terraform/frontend-ec2

terraform init
terraform plan \
  -var "ami_id=ami-xxxxxxxx" \
  -var "key_name=your-keypair-name"

terraform apply \
  -var "ami_id=ami-xxxxxxxx" \
  -var "key_name=your-keypair-name"
```