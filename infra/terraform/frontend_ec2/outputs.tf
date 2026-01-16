output "frontend_public_ip" {
  value = aws_instance.frontend.public_ip
}

output "frontend_public_dns" {
  value = aws_instance.frontend.public_dns
}
