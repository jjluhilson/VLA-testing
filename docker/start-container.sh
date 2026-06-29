
# Run the container with Host Networking and GPU Passthrough
docker run -d \
  --gpus all \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ./src/:/vla_ws/src/ \
  --name vla-image \
  vla-container:latest \
  tail -f /dev/null