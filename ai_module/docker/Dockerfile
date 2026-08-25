# Team Xiao Hei — AI module image for the CMU VLN Challenge 2026.
#
# Built on our own prebuilt, tested image from Docker Hub, so evaluation uses
# the *exact* perception stack, model weights and runtime config we validated
# rather than rebuilding the heavy torch/CUDA stack on the eval machine. The
# from-source recipe for that base is `Dockerfile.full`.
#
# This file adds the instruction-following stack on top. Everything it needs
# beyond the base is three lines, because the base already carries apt's
# opencv 4.6 and scipy 1.11.
#
#   `ros2 launch dummy_vlm dummy_vlm.launch` starts the router, which reads
#   the question and picks the stack that answers it:
#     instruction-following -> VLM grounding + lidar lift + waypoints (here)
#     numerical / reference -> exploration + perception scene graph + Gemini
#                              (scene_gemini.launch, unchanged)
FROM oel20/cmu-vln-ai-module:latest

USER root

# --- why numpy is pinned ------------------------------------------------------
# The base image resolves a pip numpy 2.5.1 into
# /usr/local/lib/python3.12/dist-packages, which shadows apt's numpy 1.26.4 in
# /usr/lib/python3/dist-packages. apt's python3-opencv is built against numpy
# 1.x, so in the base image `import cv2` raises
#
#     ImportError: numpy.core.multiarray failed to import
#
# That reads as an opencv problem and is a dependency-resolution one; it comes
# from `--ignore-installed` plus a loose `numpy>=1.26` pin in Dockerfile.full.
# The perception sidecar is unaffected either way — it lives in /opt/perc-venv
# with its own numpy 2 and torch. Pinning system python back to numpy 1.x is
# what Dockerfile.full's own comment says this environment is supposed to be
# ("this env must stay on numpy < 2"), and the grounding loop needs cv2 for the
# gnomonic remap that cuts the panorama into faces.
#
# `anthropic` is the only genuinely new dependency and pulls nothing
# numpy-shaped (httpx / pydantic / anyio / jiter).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy<2" \
        anthropic

# --- the instruction-following stack -----------------------------------------
# Deliberately a directory of plain files rather than an installed package, so
# it can be read, edited and run in place inside the container while debugging
# a failed run. `docker/run_dev.sh` bind-mounts a working copy over this path.
COPY vlm /opt/xiao_hei/vlm
ENV XIAO_HEI_VLM_DIR=/opt/xiao_hei/vlm

# --- the object-reference stack ----------------------------------------------
# `xiao_hei_vln` is baked into the base image, so until now nothing in this
# repository's `ai_module/xiao_hei_vln/` could reach a container built from
# THIS file — and this file is what `docker/compose*.yml` builds. Edits to the
# responder package were silently inert unless the base image was rebuilt and
# pushed. Copying the source over fixes that.
#
# A copy, not a `pip install`: Dockerfile.full installs the package **editable**
# (`pip install -e`) at /opt/xiao_hei_vln, so imports resolve through this
# source tree and overwriting it takes effect with no reinstall. That matters —
# a reinstall would re-resolve dependencies, and re-resolving is exactly how
# system python ended up on numpy 2 and `import cv2` started failing (see
# above). No new third-party dependency is introduced by this stack: it needs
# httpx, matplotlib, pillow and numpy, all of which the Gemini pipeline already
# pulled in, plus `anthropic`, installed above.
#
# Caveat for whoever touches this next: `pyproject.toml` is NOT re-read, so a
# new dependency or entry point still needs a base-image rebuild.
COPY xiao_hei_vln/src /opt/xiao_hei_vln/src

# PYTHONPATH is deliberately NOT extended here, and that is load-bearing.
#
# Both halves of this module ship a package called `xiao_hei_vln`, and they are
# not the same package: the one installed at /opt/xiao_hei_vln has `app`,
# `gemini` and `scene`, while the copy vendored under vlm/src carries only the
# two leaves the grounding loop needs — and its `perception.geometry` returns a
# *different* camera-lidar extrinsic (t = [0, 0.1, 0] against [0, 0, 0.1]).
# Whichever is on PYTHONPATH shadows the other for every process in the
# container: put vlm/src there and the hand-off lands on a pipeline that can no
# longer `import xiao_hei_vln.app`.
#
# `challenge_node.py` inserts its own two directories into `sys.path` at
# import, so the shadow exists only inside the instruction-following process,
# which is the only one that wants it. After the hand-off `exec`s, the
# environment is clean and the installed package resolves normally.

# --- the responder the hand-off lands on -------------------------------------
# NOT set in the base image, where `app/main.py` therefore resolves
# `RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "dummy")` to **dummy** —
# the deterministic port of the C++ dummy node, which answers numerical
# questions with a random integer and object-reference with canned fixtures.
# Both `Dockerfile.full` and `scene_gemini.launch` document the intent as the
# scene_gemini pipeline, so this is a missing variable rather than a choice.
# Stated here so the hand-off lands on the pipeline it names.
ENV XIAO_HEI_RESPONDER=scene_gemini

# --- the ament package -------------------------------------------------------
# Nothing is compiled; colcon installs the two launch files and the `dummyVLM`
# Python entry point under the names the evaluation expects.
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm"

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Both .bashrc files, because this image runs as **root**, not as the template's
# `docker` user: the base challenge image is uid 1001 with HOME=/home/docker,
# but Dockerfile.full does `USER root` for apt and never switches back. Writing
# only to /home/docker/.bashrc therefore left the overlay unsourced for the
# user that actually runs, and the documented command
#
#     docker exec -it iros2026_ai_module bash
#     ros2 launch dummy_vlm dummy_vlm.launch
#
# failed with "Package 'dummy_vlm' not found". Switching back to `docker` would
# be tidier but risks the sidecar's write access to /opt/perception (weights,
# HF_HOME, TORCH_HOME are all under it), so the runtime user is left alone and
# both home directories are covered.
RUN for rc in /home/docker/.bashrc /root/.bashrc; do \
      echo "source /etc/xiao_hei_setup.sh" >> "$rc"; \
    done

# ...and once more for shells that read neither .bashrc nor the entrypoint.
# `.bashrc` opens with the stock `[ -z "$PS1" ] && return`, so it does nothing
# for a *non-interactive* shell — and `docker exec <ctr> bash -c "ros2 launch
# dummy_vlm dummy_vlm.launch"` is exactly that, and is a plausible shape for a
# startup script we cannot see. BASH_ENV is the hook bash reads in that mode.
# The guard keeps it cheap: sourcing an overlay once per subshell is fine,
# doing it in every nested one bloats AMENT_PREFIX_PATH.
RUN printf '%s\n' \
      '# sourced by interactive shells (.bashrc) and non-interactive ones (BASH_ENV)' \
      'if [ -z "$XIAO_HEI_ROS_SOURCED" ] && [ -f /home/docker/ai_module/install/setup.bash ]; then' \
      '  source /opt/ros/jazzy/setup.bash' \
      '  source /home/docker/ai_module/install/setup.bash' \
      '  export XIAO_HEI_ROS_SOURCED=1' \
      'fi' \
    > /etc/xiao_hei_setup.sh
ENV BASH_ENV=/etc/xiao_hei_setup.sh
