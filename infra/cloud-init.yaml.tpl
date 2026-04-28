#cloud-config

package_update: true

write_files:
  - path: /etc/systemd/system/speechenhance.service
    permissions: "0644"
    content: |
      [Unit]
      Description=SpeechEnhanceApp
      After=network.target

      [Service]
      User=${admin_username}
      WorkingDirectory=/home/${admin_username}/SpeechEnhanceApp
      Environment="PATH=/home/${admin_username}/SpeechEnhanceApp/venv/bin"
      Environment="SECRET_KEY=${secret_key}"
      ExecStart=/home/${admin_username}/SpeechEnhanceApp/venv/bin/gunicorn \
        --workers 2 --bind 127.0.0.1:5000 --timeout 120 app:app
      Restart=always

      [Install]
      WantedBy=multi-user.target

  - path: /etc/nginx/sites-available/speechenhance
    permissions: "0644"
    content: |
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

  - path: /etc/sudoers.d/speechenhance
    permissions: "0440"
    content: |
      ${admin_username} ALL=(ALL) NOPASSWD: /bin/systemctl restart speechenhance

runcmd:
  # Python 3.10 (required by numba==0.58.1)
  - add-apt-repository ppa:deadsnakes/ppa -y
  - apt-get update -qq
  - apt-get install -y python3.10 python3.10-venv python3.10-distutils nginx git

  # Clone repo
  - sudo -u ${admin_username} git clone ${github_repo} /home/${admin_username}/SpeechEnhanceApp

  # Python venv + dependencies
  - sudo -u ${admin_username} python3.10 -m venv /home/${admin_username}/SpeechEnhanceApp/venv
  - sudo -u ${admin_username} /home/${admin_username}/SpeechEnhanceApp/venv/bin/pip install -q --upgrade pip
  - sudo -u ${admin_username} /home/${admin_username}/SpeechEnhanceApp/venv/bin/pip install -q -r /home/${admin_username}/SpeechEnhanceApp/requirements.txt

  # Model + uploads directories (ONNX files are copied separately via scripts/copy_models.sh)
  - mkdir -p /home/${admin_username}/SpeechEnhanceApp/src/models
  - mkdir -p /home/${admin_username}/SpeechEnhanceApp/src/uploads
  - chown -R ${admin_username}:${admin_username} /home/${admin_username}/SpeechEnhanceApp

  # nginx
  - ln -sf /etc/nginx/sites-available/speechenhance /etc/nginx/sites-enabled/speechenhance
  - rm -f /etc/nginx/sites-enabled/default
  - nginx -t
  - systemctl enable nginx
  - systemctl restart nginx

  # speechenhance gunicorn service
  - systemctl daemon-reload
  - systemctl enable speechenhance
  - systemctl start speechenhance
