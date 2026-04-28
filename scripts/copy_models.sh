#!/usr/bin/env bash
# Run this locally after initial VM provisioning to upload ONNX model weights.
# Models are excluded from git (*.onnx in .gitignore) and must be copied once.
#
# Usage:
#   chmod +x scripts/copy_models.sh
#   VM_IP=<public_ip> KEY=~/.ssh/speechapp_key.pem bash scripts/copy_models.sh

set -euo pipefail

VM_IP=${VM_IP:?'Set VM_IP before running: export VM_IP=<public_ip>'}
VM_USER=${VM_USER:-azureuser}
KEY=${KEY:?'Set KEY before running: export KEY=~/.ssh/speechapp_key.pem'}
REMOTE_DIR="~/SpeechEnhanceApp/src/models"
SCP="scp -i $KEY -o StrictHostKeyChecking=no"

echo "Copying ONNX models to $VM_USER@$VM_IP:$REMOTE_DIR ..."

# 16 kHz models
$SCP src/models/upsample16_gan_500.onnx      "$VM_USER@$VM_IP:$REMOTE_DIR/"
$SCP src/models/upsample16_audiounet_500.onnx "$VM_USER@$VM_IP:$REMOTE_DIR/"

# 48 kHz models
$SCP src/models/upsample48_gan_500.onnx      "$VM_USER@$VM_IP:$REMOTE_DIR/"
$SCP src/models/upsample48_audiounet_500.onnx "$VM_USER@$VM_IP:$REMOTE_DIR/"

echo "Done. Restart the service on the VM:"
echo "  ssh -i $KEY $VM_USER@$VM_IP 'sudo systemctl restart speechenhance'"
