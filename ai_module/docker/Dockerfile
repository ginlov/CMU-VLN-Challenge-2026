# Team Xiao Hei — AI module image for the CMU VLN Challenge 2026.
#
# This Dockerfile intentionally just pulls our prebuilt, tested image from
# Docker Hub, so evaluation uses the *exact* image we validated (perception
# stack, scene_gemini responder, model weights, and runtime config all baked
# in) rather than rebuilding the heavy torch/CUDA stack on the eval machine.
#
#   `ros2 launch dummy_vlm dummy_vlm.launch`  brings up the whole pipeline:
#   frontier exploration -> perception scene graph (YOLO-World + SAM) -> Gemini.
#
# The full, from-source build recipe (ROS base + isolated perception venv +
# our package + weight downloads) is kept alongside as `Dockerfile.full` for
# transparency and reproducibility:
#
#   docker build -t oel20/cmu-vln-ai-module:latest -f docker/Dockerfile.full .
#
# Rebuild that, push it, and this one-liner picks it up on the next pull.

FROM oel20/cmu-vln-ai-module:latest
