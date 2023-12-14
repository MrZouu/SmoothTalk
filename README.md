# SmoothTalk
![Language](https://img.shields.io/badge/Language-Python-f2cb1b)
![Libraries](https://img.shields.io/badge/Libraries-TensorFlow-FF6F00)
![Libraries](https://img.shields.io/badge/Libraries-PyTorch-00cf2c)

Sign language interpreter, to switch from a video signal to a corresponding word.
This machine-learning project, powered by TensorFlow, aims to develop a sign language recognition system.

⚠️This project is the result of a common work, made for school with a deadline and is still under development

# GPU management
##  Google Colab - Tensorflow
For this project, Google colab is a very useful tool for processing our data in our machine learning pipeline.  
In fact, Colab includes a gpu, enabling us to process our data more quickly and simply.  
- ## Usage
* **Import code** (jupyter file) into colab
* **Upload a dataset in zip format to the Google Drive** - associated with the google account used on Colab.
* **Change path in code and log in**
* **Respect the tree structure**

## Cuda Toolkit - Pytorch
An alternative to Google Colab is the Cuda Toolkit.
Cuda is used in particular with Pytorch. Here, you can use your own computer gpu.
The advantage is that we are not limited by the free version of Google Colab.
- ## Usage
* **Python** - We recommend installing the [3.10.7](https://www.python.org/downloads/release/python-3107/) Version
* **Cuda Toolkit** - Install version [11.7.0](https://developer.nvidia.com/cuda-toolkit-archive) - exe installer
* **Pytorch** - Version 1.13.1. Install the version corresponding to the CUDA version on the [website](https://pytorch.org/)
- Command path to enter in cmd : "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117"
* **Installation check** : cmd -> python -> import torch -> torch.cuda.device_count() -> Should display "1"

#  Credits
* [**Lorenzo**](https://github.com/MrZouu) : Co-creator of the project.
* [**Sc0pziion**](https://github.com/sc0pziion) : Co-creator of the project.
* [**Clement Auray**](https://github.com/Clementauray) : Co-creator of the project.
* [**Evann Ali-Yahia**](https://github.com/EvannAyh) : Co-creator of the project.
* [**Yassine Drici**](https://github.com/Yasssdz) : Co-creator of the project.
* [**ThhoommaassR**](https://github.com/ThhoommaassR) : Co-creator of the project.
