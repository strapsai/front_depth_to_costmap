# Base Image for JetPack 6.1 (Ubuntu 22.04)
# Supported Versions:
# - CUDA: 12.6
# - PyTorch: 2.4.0
# - OpenCV: 4.10.0
# - TorchAudio: 2.4.0
# - TorchVision: 0.19.0



ARG base_image
# Base image with CUDA support
FROM ${base_image}

# Install necessary dependencies, including gedit and ping

ARG DEBIAN_FRONTEND=noninteractive
    
RUN apt-get update \
     && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
     && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
     && dpkg-reconfigure --frontend noninteractive tzdata \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1-mesa-glx libx11-6 x11-apps \
    apt-utils \
    libopenblas-dev \
    software-properties-common \
    gedit \
    iputils-ping \
    tmux \
    nano \
    vim \
    net-tools \
    less \
    libgl-dev\
    wget \ 
    unzip \
    make \
    cmake \
    xz-utils \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/*

CMD ["bash"]

#NOTE(kabirkedia: 12/21/2024): this dockerfile will not not build the opencv support for c++ natively. 

#There is an error. I have fixed the error and the image is available on dockerhub