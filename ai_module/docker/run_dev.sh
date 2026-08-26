#!/bin/bash
#
# Same as run.sh, but with the drive loop bind-mounted from the working tree.
#
# The loop is a directory of plain files inside the image rather than an
# installed package precisely so it can be changed without a rebuild: edit
# ai_module/vlm/... on the host, re-run `ros2 launch vlm vlm.launch`
# in the container, and the change is live. A rebuild takes minutes and there
# are ten of those in a question.
#
# Not for submission runs — use run.sh, which runs what was actually built.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VLM_DIR="$( cd "$SCRIPT_DIR/../vlm" && pwd )"

if command -v nvidia-smi &> /dev/null; then
  GPU_FLAGS="--gpus all"
else
  GPU_FLAGS=""
fi

xhost + || true

docker run $GPU_FLAGS -it --rm --privileged \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=/tmp/.docker.xauth \
  -e ANTHROPIC_API_KEY \
  -e XIAO_HEI_CLASSIFY \
  -e XIAO_HEI_IMAGE_TOPIC \
  -e XIAO_HEI_BUDGET_S \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /etc/localtime:/etc/localtime:ro \
  -v "$VLM_DIR":/opt/xiao_hei/vlm \
  --network=host \
  iros2026/ai_module:latest \
  "${@:-bash}"
