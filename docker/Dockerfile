FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + graphics libraries for MuJoCo rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libegl1 \
    libgles2 \
    libglfw3 \
    libglew-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /vla_ws

# Install Gymnasium with robotics dependencies and MuJoCo
RUN pip3 install --no-cache-dir --break-system-packages \
    gymnasium \
    gymnasium-robotics \
    mujoco \
    numpy

COPY src /vla_ws/src

CMD ["bash"]