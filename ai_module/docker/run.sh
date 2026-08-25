#!/bin/bash

# ANTHROPIC_API_KEY is required, and it is the only key this module needs:
# every question type — classification, instruction following, numerical and
# object reference — is answered by calling the Claude API. No key is baked
# into the image.
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "warning: ANTHROPIC_API_KEY is not set in this shell; every question" >&2
  echo "         type will fail. See ai_module/README.md." >&2
fi

# Use GPU flags if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
  GPU_FLAGS="--gpus all"
else
  GPU_FLAGS=""
fi

xhost +

docker run $GPU_FLAGS -it --rm --privileged \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e ANTHROPIC_API_KEY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/localtime:/etc/localtime:ro \
  --network=host \
  iros2026/ai_module:latest
