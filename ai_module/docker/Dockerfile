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
#   `ros2 launch vlm vlm.launch` starts the router, which reads
#   the question and picks the stack that answers it:
#     instruction-following -> VLM grounding + lidar lift + waypoints (here)
#     numerical             -> answered in that same process
#     object reference      -> question-directed nav + perception scene graph
#                              + Claude (scene_claude.launch)
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
#
# --- why anthropic is pinned, and pinned to exactly this ----------------------
# Unpinned, this line resolved to a different SDK depending on the day the
# image was built, and the two are not interchangeable:
#
#   anthropic <  1.0.0   depends on httpx  (plain, what is installed here)
#   anthropic >= 1.0.0   depends on httpx2 (a fork, released 2026-08-20)
#
# httpx2's brotli decoder guards against decompression bombs by calling
#
#     self._decompress(data, output_buffer_limit=...)
#
# and that keyword argument only exists in Google `brotli` >= 1.2.0. The base
# image carries apt's python3-brotli 1.1.0, whose `process()` is a C method
# taking no keyword arguments, so every response the API compresses dies with
#
#     TypeError: process() takes no keyword arguments
#
# which httpx2 wraps as a transport error and the SDK re-wraps as
# `APIConnectionError: Connection error.` — a message that sends you to look at
# the network and the API key, neither of which is involved: the request goes
# out fine and the failure is in decoding the reply. Upgrading brotli instead
# would mean pip overwriting an apt package, i.e. exactly the numpy trap above.
#
# 0.122.0 is not "a version that works", it is the version every run in this
# repo's results was produced on (verified in the images on the run hosts).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "numpy<2" \
        "anthropic==0.122.0"

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
# Wipe first, because COPY overwrites but never deletes. Without this, a
# module deleted from this repository lives on in the image for as long as the
# base image carries it — `xiao_hei_vln/dummy`, `gemini` and `scene_gemini` all
# outlived their removal that way, so `dummy` was still importable in an image
# built from a tree that no longer had it. The package is a plain source tree
# behind an editable install, so replacing it wholesale is safe and makes the
# image match the repository exactly.
RUN rm -rf /opt/xiao_hei_vln/src/xiao_hei_vln
COPY xiao_hei_vln/src /opt/xiao_hei_vln/src

# --- perception: use the validated SAM 2.1 Hiera-Large -----------------------
# The base image bakes in SAM Hiera-Tiny (Dockerfile.full) and its perception
# source at /opt/perception, and the base override above does NOT copy
# perception/ — so, exactly like the xiao_hei_vln case, editing
# ai_module/perception/ would be silently inert on THIS build. Fetch the Large
# weight into the models dir the base already provisioned, then overwrite the
# sidecar source so pipeline.py loads it. The Tiny weight is left in place
# (harmless; it sits in a base layer that a delete here would not reclaim).
RUN curl -fsSL -o /opt/perception/models/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
COPY perception/server.py perception/pipeline.py perception/geometry.py perception/__init__.py \
     /opt/perception/

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
# `RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "scene_claude")` —
# the deterministic port of the C++ dummy node, which answers numerical
# questions with a random integer and object-reference with canned fixtures.
# a missing variable rather than a choice. Stated here so the hand-off lands on
# the pipeline `scene_claude.launch` names. The launch file sets it too; this is
# the belt to that pipeline's braces, and covers a bare `xiao-hei-vlm`.
ENV XIAO_HEI_RESPONDER=scene_claude

# --- the ament package -------------------------------------------------------
# Nothing is compiled; colcon installs the launch files and the `vlmNode`
# Python entry point under the names the evaluation expects.
COPY src/vlm /home/docker/ai_module/src/vlm
COPY src/dummy_vlm /home/docker/ai_module/src/dummy_vlm
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select vlm dummy_vlm"

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Both .bashrc files, because this image runs as **root**, not as the template's
# `docker` user: the base challenge image is uid 1001 with HOME=/home/docker,
# but Dockerfile.full does `USER root` for apt and never switches back. Writing
# only to /home/docker/.bashrc therefore left the overlay unsourced for the
# user that actually runs, and the documented command
#
#     docker exec -it iros2026_ai_module bash
#     ros2 launch vlm vlm.launch
#
# failed with "Package 'vlm' not found". Switching back to `docker` would
# be tidier but risks the sidecar's write access to /opt/perception (weights,
# HF_HOME, TORCH_HOME are all under it), so the runtime user is left alone and
# both home directories are covered.
RUN for rc in /home/docker/.bashrc /root/.bashrc; do \
      echo "source /etc/xiao_hei_setup.sh" >> "$rc"; \
    done

# ...and once more for shells that read neither .bashrc nor the entrypoint.
# `.bashrc` opens with the stock `[ -z "$PS1" ] && return`, so it does nothing
# for a *non-interactive* shell — and `docker exec <ctr> bash -c "ros2 launch
# vlm vlm.launch"` is exactly that, and is a plausible shape for a
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

# --- the API key -------------------------------------------------------------
# The key is NOT in this repository and NOT in any image we publish. It is
# supplied with the submission and injected here at build time:
#
#     docker compose -f compose_gpu.yml build \
#         --build-arg ANTHROPIC_API_KEY=<the key from the submission>
#     docker compose -f compose_gpu.yml up -d
#
# Build time rather than run time, because `docker/compose*.yml` declares no
# `environment:` block for this service and lives outside `ai_module/`, which
# the challenge README says is the only folder a submission may change. So
# nothing exported on the host reaches the container, and a `docker exec`
# export is lost the moment the container is restarted — which evaluation does
# for every question ("the system will be relaunched for each language command
# tested"). An ARG baked to ENV survives `up`, `restart` and `run` alike.
#
# Last in the file so that changing the key rebuilds this layer only.
#
# Left empty when the argument is not passed. That is deliberate and not a
# fallback: `challenge_node.py` checks for it at boot and exits with a message
# naming the variable, rather than driving somewhere first and failing on the
# call that was supposed to decide where to drive.
ARG ANTHROPIC_API_KEY=""
ENV ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# The base image on Docker Hub predates the move to a single model API and
# still carries a `XIAO_HEI_GEMINI_API_KEY` in its ENV, which `FROM` inherits.
# That key has been revoked, so this is tidiness rather than exposure — but a
# submitted image should not ship a key-shaped string that a grader has to
# reason about, and blanking it here is cheaper than rebuilding and re-pushing
# a 10 GB base. Nothing reads it: the Gemini SDK is not installed and
# `ask_gemini()` refuses.
#
# Dockerfiles cannot delete an inherited variable, only empty it. Empty is
# enough — the value is what mattered.
ENV XIAO_HEI_GEMINI_API_KEY=""
