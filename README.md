# Speech Enhancement

## Table of Contents

- [General Info](#general-info)
- [Usage](#usage)
- [Deployment](#deployment)
- [Technologies](#technologies)

## General Info

A web app demonstrating ML models for speech denoising and bandwidth extension (16 kHz → 48 kHz). Inference is performed using ONNX format models. Accepts audio in WAV or FLAC format.

## Usage

**Local machine:**

```bash
pip install -r requirements.txt
python app.py
```

Place model files in `./src/models/`:

```
src/models/denoise/denoise_gan.onnx
src/models/upsample/upsample16_gan_500.onnx
src/models/upsample/upsample48_gan_500.onnx
```

App will be available at `http://localhost:5000`.

## Deployment

The app is deployed on an **Azure Linux VM** with GitHub Actions CI/CD and is accessible at [https://speechenhance.com](https://speechenhance.com).

### Stack

- **Azure VM** (Ubuntu 22.04) — hosts the app
- **Gunicorn** — WSGI server running on `127.0.0.1:5000`
- **Nginx** — reverse proxy on port 80/443
- **Let's Encrypt** — free SSL certificate via certbot
- **Cloudflare** — DNS, proxy, and DDoS protection
- **GitHub Actions** — CI/CD pipeline (build → test → deploy)

### CI/CD Pipeline

Every push to `main` triggers three sequential jobs:

1. **Build** — installs Python 3.10 and dependencies
2. **Test** — runs `pytest tests/` (deploy is blocked if tests fail)
3. **Deploy** — SSH into VM, pulls latest code, restarts the service

ONNX model files (~0.5 GB) are not stored in git. They are copied to the VM once manually via `scp` and remain there permanently.

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `VM_HOST` | Azure VM public IP |
| `VM_USER` | VM username (`azureuser`) |
| `VM_SSH_KEY` | SSH private key (full `.pem` contents) |

For full step-by-step deployment instructions see `SpeechEnhanceApp_Deployment_Guide.docx`.

## Technologies

- Python 3.10
- Flask 3.0.0
- Gunicorn 21.2.0
- Jinja2 3.1.2
- librosa 0.10.1
- scikit-learn 1.3.2
- onnxruntime
