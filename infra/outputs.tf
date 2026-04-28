output "public_ip" {
  description = "VM public IP — use this for VM_HOST GitHub secret and Cloudflare DNS"
  value       = azurerm_public_ip.pip.ip_address
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh -i ${var.ssh_public_key_path} ${var.admin_username}@${azurerm_public_ip.pip.ip_address}"
  sensitive   = false
}
