variable "location" {
  description = "Azure region"
  default     = "northeurope"
}

variable "resource_group_name" {
  description = "Resource group name"
  default     = "speechenhance-rg"
}

variable "vm_size" {
  description = "VM SKU (minimum B2s for ONNX inference)"
  default     = "Standard_B2s"
}

variable "admin_username" {
  description = "Linux admin user on the VM"
  default     = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Path to local SSH public key file"
  default     = "~/.ssh/id_rsa.pub"
}

variable "secret_key" {
  description = "Flask SECRET_KEY — generate with: python3 -c \"import secrets; print(secrets.token_hex(32))\""
  sensitive   = true
}

variable "github_repo" {
  description = "HTTPS URL of the GitHub repository"
  default     = "https://github.com/msilaev/SpeechEnhanceApp.git"
}
