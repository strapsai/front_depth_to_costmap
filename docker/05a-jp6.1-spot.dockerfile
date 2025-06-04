# Spot Driver Image for JetPack 6.1 (Ubuntu 22.04)
# Builds on top of spot driver dockerfile
# - Spot Driver: 4.1
# - Open3D: 0.18.0
# - Microstrain Inertial Driver

ARG base_image
# Base image with CUDA/ROS2 support
FROM ${base_image}

ARG DEBIAN_FRONTEND=noninteractive

# ROS2 GPG Key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

#Some additional ros dependencies I may have missed
RUN apt-get update --no-install-recommends \
 && apt-get install -y \
    unzip \
    build-essential \
    gir1.2-gst-plugins-bad-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gudev-1.0 \
    gstreamer1.0-alsa \
    gstreamer1.0-gtk3 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-qt5 \
    gstreamer1.0-tools \
    libatlas-base-dev \
    libcdio19 \
    libdw-dev \
    libelf-dev \
    libeigen3-dev \
    libfmt-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-dev \
    libgudev-1.0-dev \
    libmpeg2-4 \
    libopencore-amrnb0 \
    libopencore-amrwb0 \
    libopenexr-dev \
    liborc-0.4-dev \
    liborc-0.4-dev-bin \
    libparmetis-dev \
    libpcap-dev \
    libqt5waylandclient5 \
    libqt5x11extras5 \
    libsidplay1v5 \
    libsuitesparse-dev \
    libunwind-dev \
    libx11-xcb-dev \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    ros-humble-can-msgs \
    ros-humble-diagnostic-updater \
    ros-humble-domain-bridge \
    ros-humble-geographic-msgs \
    ros-humble-nmea-msgs \
    ros-humble-robot-localization \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rviz2 \
    ros-humble-serial-driver \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    ros-humble-turtle-tf2-py \
    ros-humble-turtlesim

#Below are specific dependecnies and binaries for spot  
# Install ROS dependencies
# TODO(jschornak-bdai): use rosdep to install these packages by parsing dependencies listed in package.xml
ARG ROS_DISTRO=humble
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ros-$ROS_DISTRO-joint-state-publisher-gui \
    ros-$ROS_DISTRO-xacro \
    ros-$ROS_DISTRO-tl-expected \
    ros-$ROS_DISTRO-ros2-control \
    ros-$ROS_DISTRO-ros2-controllers \
    ros-$ROS_DISTRO-tf-transformations \
    ros-$ROS_DISTRO-depth-image-proc \
    ros-$ROS_DISTRO-controller-interface \
    ros-$ROS_DISTRO-forward-command-controller \
    ros-$ROS_DISTRO-bondcpp \
    ros-$ROS_DISTRO-bond \
    ros-$ROS_DISTRO-smclib \
    ros-$ROS_DISTRO-bondpy \
    clang-tidy \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install the dist-utils
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3-distutils \
    python3-apt \
    python3-rpi.gpio \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ARG ARCH=$CPU_ARCH
# CMD if [ "$ARCH_T" = "x86"] ; then ARG ARCH="amd64"; else echo ARG ARCH="arm64" ; fi

ENV ARCH="arm64"
ARG SDK_VERSION="4.1.0"
ARG MSG_VERSION="${SDK_VERSION}-4"

#we need this version of setuptools
RUN pip3 install --force-reinstall -v "setuptools==59.6.0"
RUN pip3 install --no-cache-dir \
    aiortc==1.5.0 \
    bosdyn-api==4.1.0 \
    bosdyn-choreography-client==4.1.0 \
    bosdyn-client==4.1.0  \
    bosdyn-core==4.1.0 \
    bosdyn-mission==4.1.0 \
    grpcio==1.59.3 \
    image==1.5.33 \
    inflection==0.5.1 \
    protobuf==4.22.1 \
    pytest==7.3.1 \
    pytest-cov==4.1.0 \
    pytest-xdist==3.5.0 \
    "pyyaml>=6.0" 

#additional dependencies
RUN pip3 install --no-cache-dir \
    ipdb \
    ipython \
    nvitop==1.3.2 \
    Jetson.GPIO

# cv_bridge needs to be built from source as a ROS package in our ROS workspace.
# It is in the vision_opencv repo and the humble branch. 
# RUN pip3 install --no-cache-dir cv-bridge
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends \
#     ros-humble-vision-opencv \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

# Install bosdyn_msgs - automatic conversions of BD protobufs to ROS messages
RUN wget -q -O /tmp/ros-humble-bosdyn_msgs_${MSG_VERSION}-jammy_${ARCH}.run https://github.com/bdaiinstitute/bosdyn_msgs/releases/download/${MSG_VERSION}/ros-humble-bosdyn_msgs_${MSG_VERSION}-jammy_${ARCH}.run
RUN chmod +x /tmp/ros-humble-bosdyn_msgs_${MSG_VERSION}-jammy_${ARCH}.run
RUN yes | /tmp/ros-humble-bosdyn_msgs_${MSG_VERSION}-jammy_${ARCH}.run  --nox11
RUN rm /tmp/ros-humble-bosdyn_msgs_${MSG_VERSION}-jammy_${ARCH}.run

# Install spot-cpp-sdk
RUN wget -q -O /tmp/spot-cpp-sdk_${SDK_VERSION}_${ARCH}.deb https://github.com/bdaiinstitute/spot-cpp-sdk/releases/download/v${SDK_VERSION}/spot-cpp-sdk_${SDK_VERSION}_${ARCH}.deb
RUN dpkg -i /tmp/spot-cpp-sdk_${SDK_VERSION}_${ARCH}.deb
RUN rm /tmp/spot-cpp-sdk_${SDK_VERSION}_${ARCH}.deb

RUN wget -q -O /tmp/protoc-29.0-rc-3-linux-aarch_64.zip https://github.com/protocolbuffers/protobuf/releases/download/v29.0-rc3/protoc-29.0-rc-3-linux-aarch_64.zip

RUN rm -rf /usr/local/bin/protoc /usr/local/include/google /usr/local/lib/libproto*

RUN unzip -o /tmp/protoc-29.0-rc-3-linux-aarch_64.zip -d /usr/local

# open3d. open3d tends to upgrade numpy which causes many issues later. Also, blinker failed to be
# uninstalled by open3d. blinker==1.9.0 is the current version resolved by pip3 when trying to
# install open3d==0.18.0.
RUN pip3 install --no-cache-dir --ignore-installed blinker==1.9.0
RUN NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)") && \
    pip3 install --no-cache-dir \
      numpy==${NUMPY_VERSION} \
      open3d==0.18.0

# Dependencies for the GPS RTK driver
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ros-humble-microstrain-inertial-driver \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


# Dependencies for the  GStreamer Camera
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
               gstreamer1.0-tools \
               gstreamer1.0-x \
               gstreamer1.0-plugins-base \
               gstreamer1.0-plugins-good \
               gstreamer1.0-plugins-bad \
               gstreamer1.0-plugins-ugly libxml2 libpcap0.8 libaudit1 libnotify4 \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

# Install the ROS package for the Foxglove Bridge: Required By Basti
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-foxglove-bridge \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*      
         

