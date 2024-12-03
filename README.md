# Speech enhancement
<!-- ![Denoising and band extension of speech](SoundRecScreenshot.jpg){width=50%} -->


## Table of contents
* [General info](#general-info)
* [Deployment](#deployment)
* [Technologies](#technologies)
* [Setup](#setup)

## General info

The goal of this project is to develop a simple app demonstrating ML models to denosie and expand bandwidth (from 16 KHz to 48 KHz) of speech signals. 
This app uses inference with the help of  models in ONNX format.  
Be ready to upload audio in WAV or FLAC formats. 

## Usage
On the local machine:

`pip install -r requirements.txt`

Download models and put then folder ```models```

[denoise.onnx] (https://drive.google.com/file/d/1gpITH4NrutQGQfBuakMW6odlVduCz2F2/view?usp=sharing)
[upsample48.onnx] (https://drive.google.com/file/d/1N39IgLdtxbFRSSXu4Wqd3Mn7AaZO2tax/view?usp=sharing)

Run app from the root folder

`python app.py`

You will then be able to access it at localhost:5000

## Deployment 

The application can deployed at AWS Lightsail containers using Docker and this [tutorial] (https://aws.amazon.com/blogs/aws/lightsail-containers-an-easy-way-to-run-your-containers-in-the-cloud/). 

## Technologies
Project is created with:
* Python 3.11.0 
* Flask==3.0.0
* Jinja2==3.1.2
* gunicorn==21.2.0
* librosa==0.10.1
* scikit-learn==1.3.2
* onnx==1.17.0
* onnxruntime==1.20.1

It was tested in a browser 
* Microsoft Edge Version 114.0.1823.43 (Official build) (64-bit)

