#FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
#FROM --platform=linux/amd64 ubuntu:20.04 AS builder-image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 AS builder-image

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt install software-properties-common -y 
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt-get install -y --no-install-recommends python3.8-venv
RUN apt-get install libpython3.8-de -y
RUN apt-get install python3.8-dev -y
RUN apt-get install build-essential -y
RUN apt-get clean



RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    libopenjp2-7-dev libopenjp2-tools \
    openslide-tools \
    libgl1
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
    
RUN echo `python3 --version`
COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip install evalutils==0.3.1 fastai==1.0.61 hydra-core==1.3.2

RUN pip install numpy==1.19.5 object_detection_fastai==0.0.10 omegaconf==2.2.3 opencv_python_headless==4.5.5.64 openslide_python==1.1.2 pandas==1.1.2
RUN pip install Pillow==9.4.0 
RUN pip install PyYAML==6.0.1 
RUN pip install scikit_learn==1.0 
RUN pip install SimpleITK==2.2.1 
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pyqt5
RUN pip install SlideRunner==2.0
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tqdm==4.49.0
RUN pip install opencv-python==4.5.1.48
RUN pip install pytest-shutil
RUN pip install boto3==1.34.69
#RUN pip install nibabel#not needed

RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

WORKDIR /app
COPY . .
#COPY "./python.py" .

#ENTRYPOINT  ["bash","entrypoint.sh"]
ENTRYPOINT  ["python3","app.py"]
