#!/bin/bash

# ANTHROPIC_API_KEY is required for instruction-following questions: that
# stack answers by calling the Claude API. It is the only key passed in from
# outside — the Gemini key used by the numerical / object-reference pipeline
# is baked into the image as ENV (see docker/Dockerfile.full).
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "warning: ANTHROPIC_API_KEY is not set in this shell; instruction-" >&2
  echo "         following questions will fail. See ai_module/README.md." >&2
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
