#!/usr/bin/env bash
# Run on the VM after first SSH connection:
#   bash <(curl -fsSL https://raw.githubusercontent.com/msilaev/SpeechEnhanceApp/main/scripts/setup_vm.sh)
# Or copy and run:
#   scp -i ~/.ssh/speechprocess_key.pem scripts/setup_vm.sh azureuser@<IP>:~/
#   ssh -i ~/.ssh/speechprocess_key.pem azureuser@<IP> "SECRET_KEY=<key> bash setup_vm.sh"

set -euo pipefail

REPO_URL="https://github.com/msilaev/SpeechEnhanceApp.git"
APP_DIR="$HOME/SpeechEnhanceApp"
VENV="$APP_DIR/venv"
SECRET_KEY="${SECRET_KEY:?'Set SECRET_KEY before running: export SECRET_KEY=<value>'}"

echo "=== 1. System packages ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -qq
sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils nginx git

echo "=== 2. Clone repo ==="
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git fetch origin && git reset --hard origin/main
else
    git clone "$REPO_URL" "$APP_DIR"
fi

echo "=== 3. Python venv ==="
python3.10 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "=== 4. Directories ==="
mkdir -p "$APP_DIR/src/models"
mkdir -p "$APP_DIR/src/uploads"

echo "=== 5. Systemd service ==="
sudo tee /etc/systemd/system/speechenhance.service > /dev/null <<EOF
[Unit]
Description=SpeechEnhanceApp
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV/bin"
Environment="SECRET_KEY=$SECRET_KEY"
ExecStart=$VENV/bin/gunicorn --workers 2 --bind 127.0.0.1:5000 --timeout 120 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo "=== 6. Sudoers (CI/CD restart) ==="
echo "$USER ALL=(ALL) NOPASSWD: /bin/systemctl restart speechenhance" \
    | sudo tee /etc/sudoers.d/speechenhance > /dev/null
sudo chmod 440 /etc/sudoers.d/speechenhance

echo "=== 7. Nginx ==="
sudo tee /etc/nginx/sites-available/speechenhance > /dev/null <<'EOF'
server {
    listen 80;
    client_max_body_size 50M;
    location / {
        proxy_pass         http://127.0.0.1:5000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/speechenhance /etc/nginx/sites-enabled/speechenhance
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "=== 8. Start service ==="
sudo systemctl daemon-reload
sudo systemctl enable speechenhance
sudo systemctl start speechenhance
sudo systemctl status speechenhance --no-pager

echo ""
echo "Setup complete. Copy ONNX models next:"
echo "  scp -i ~/.ssh/speechprocess_key.pem src/models/*.onnx azureuser@<IP>:~/SpeechEnhanceApp/src/models/"
